#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np
import tensorrt as trt

import pycuda.autoinit  # crea el contexto CUDA automáticamente
import pycuda.driver as cuda

from openni import openni2


# ---------------------------
# Utilidades: IoU, NMS y dibujo
# ---------------------------

def iou(box, boxes):
    """
    box:  (4,)  -> [x1, y1, x2, y2]
    boxes: (N,4)
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter + 1e-6

    return inter / union


def nms(boxes, scores, iou_thresh=0.5):
    """
    boxes:  (N,4)
    scores: (N,)
    devuelve índices de las cajas que se mantienen
    """
    if len(boxes) == 0:
        return []

    idxs = scores.argsort()[::-1]  # de mayor a menor
    keep = []

    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        ious = iou(boxes[i], boxes[rest])
        idxs = rest[ious <= iou_thresh]

    return keep


# <<< NUEVO: dibujar multi-clase
def draw_boxes_multiclass(frame, boxes_xyxy, scores, class_ids, class_names):
    """
    boxes_xyxy: (N,4)
    scores:     (N,)
    class_ids:  (N,)
    """
    for box, score, cid in zip(boxes_xyxy, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        if 0 <= cid < len(class_names):
            cname = class_names[cid]
        else:
            cname = f"class_{cid}"
        label = f"{cname} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


# <<< NUEVO: cargar nombres de clase desde .txt
def load_class_names(path):
    try:
        with open(path, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        if not names:
            raise ValueError("El fichero está vacío")
        print(f"[OK] Cargadas {len(names)} clases desde {path}: {names}")
        return names
    except Exception as e:
        print(f"[WARN] No se pudieron cargar clases desde {path}: {e}")
        print("[WARN] Usando clase por defecto: ['objeto']")
        return ["objeto"]


# ---------------------------
# Carga del engine TRT
# ---------------------------

def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        raise RuntimeError("No se pudo deserializar el engine")
    print(f"[OK] Engine cargado desde: {engine_path}")
    return engine


# ---------------------------
# Bucle de inferencia con Xtion + ventana
# ---------------------------

def main(args):
    engine = load_engine(args.engine)
    context = engine.create_execution_context()

    # <<< cargamos clases
    class_names = load_class_names(args.classes)
    num_classes_txt = len(class_names)

    # Descubrir nombres de tensores de entrada/salida
    io_count = engine.num_io_tensors
    input_names = []
    output_names = []

    print("\n=== TENSORES DEL ENGINE ===")
    for i in range(io_count):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        print(f"Tensor {i}: {name}")
        print(f" - modo  = {mode}")
        print(f" - shape = {shape}")
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    if len(input_names) != 1 or len(output_names) != 1:
        raise RuntimeError("Este script asume 1 input y 1 output en el engine.")

    input_name = input_names[0]    # debería ser "images"
    output_name = output_names[0]  # debería ser "output0"

    # <<< sacamos H,W reales del engine
    in_shape = engine.get_tensor_shape(input_name)  # (1,3,640,640)
    _, _, net_h, net_w = in_shape
    print(f"\n[DEBUG] Entrada del modelo: {in_shape} (H={net_h}, W={net_w})")

    # Reservamos memoria en GPU para input y output
    input_shape = (1, 3, net_h, net_w)
    input_nbytes = int(np.prod(input_shape) * np.float32().nbytes)

    out_shape = engine.get_tensor_shape(output_name)  # ej: (1, 5, N) o (1, C, N)
    print(f"[DEBUG] out_shape = {out_shape}")
    output_nbytes = int(np.prod(out_shape) * np.float32().nbytes)

    d_input = cuda.mem_alloc(input_nbytes)
    d_output = cuda.mem_alloc(output_nbytes)

    # Buffers host
    h_input = np.empty(input_shape, dtype=np.float32)
    h_output = np.empty(out_shape, dtype=np.float32)

    stream = cuda.Stream()

    # Configurar el contexto
    context.set_input_shape(input_name, input_shape)
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # === XTION / OPENNI: inicializamos cámara ===
    OPENNI2_PATH = "/usr/lib/aarch64-linux-gnu"
    openni2.initialize(OPENNI2_PATH)

    dev = openni2.Device.open_any()
    rgb = dev.create_color_stream()
    rgb.start()

    vm = rgb.get_video_mode()
    width, height = vm.resolutionX, vm.resolutionY
    print(f"\n=== XTION INICIALIZADA ===")
    print(f"Resolución RGB: {width} x {height}\n")

    conf_thres = args.conf_thres
    iou_thres = args.iou_thres

    print("=== INICIANDO INFERENCIA SOBRE XTION (CON VENTANA) ===")
    print("Pulsa 'q' para salir.\n")

    prev_time = time.time()
    fps = 0.0
    frame_id = 0

    try:
        while True:
            # Leer frame de la Xtion
            frame_oni = rgb.read_frame()
            data = np.frombuffer(frame_oni.get_buffer_as_uint8(), dtype=np.uint8)
            img_rgb = data.reshape((height, width, 3))
            frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # BGR

            frame_id += 1
            orig_h, orig_w = frame.shape[:2]

            # Preprocesado: BGR -> RGB, resize, normalizar, CHW
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = np.expand_dims(img, 0).copy()  # (1,3,H,W)
            h_input[...] = img

            # Copiamos al device
            cuda.memcpy_htod_async(d_input, h_input, stream)

            # Ejecutamos
            context.execute_async_v3(stream_handle=stream.handle)

            # Copiamos de vuelta la salida
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()

            # Decodificar la salida
            out = h_output[0]           # quitamos batch -> (C, N) o (N, C)
            if out.ndim != 2:
                print("[ERROR] Salida no es 2D, shape =", out.shape)
                break

            # Normalizar para trabajar como (C, N)
            if out.shape[0] < out.shape[1]:
                # suponemos (C, N)
                out_cn = out
            else:
                # suponemos (N, C)
                out_cn = out.T

            C, N = out_cn.shape

            # Caso 1: 1 sola clase: (5, N) -> [x, y, w, h, score]
            if C == 5:
                boxes_xywh = out_cn[0:4, :]     # (4, N)
                scores_all = out_cn[4, :]       # (N,)
                class_ids_all = np.zeros(N, dtype=np.int32)  # todo clase 0

            # Caso 2: multi-clase: (4+nc, N) -> [x,y,w,h, cls0, cls1, ...]
            elif C > 5:
                boxes_xywh = out_cn[0:4, :]     # (4, N)
                cls_scores = out_cn[4:, :]      # (nc, N)
                # mejor clase para cada caja
                class_ids_all = np.argmax(cls_scores, axis=0)          # (N,)
                scores_all = cls_scores[class_ids_all, np.arange(N)]   # (N,)

                # aviso si nc no cuadra con clases.txt
                nc = cls_scores.shape[0]
                if nc != num_classes_txt:
                    # solo avisar una vez
                    if frame_id == 1:
                        print(f"[WARN] El modelo tiene {nc} clases, pero en {args.classes} hay {num_classes_txt}.")
            else:
                print("[ERROR] Menos de 5 canales en la salida, shape =", out_cn.shape)
                break

            # Filtro de confianza
            mask = scores_all > conf_thres
            boxes_xywh = boxes_xywh[:, mask]
            scores_f = scores_all[mask]
            class_ids_f = class_ids_all[mask]

            num_before_nms = boxes_xywh.shape[1]
            num_after_nms = 0
            boxes_xyxy = np.empty((0, 4), dtype=np.float32)
            class_ids_nms = np.empty((0,), dtype=np.int32)
            scores_nms = np.empty((0,), dtype=np.float32)

            if num_before_nms > 0:
                # xywh -> xyxy en espacio de red (net_w, net_h)
                x_c, y_c, w, h = boxes_xywh
                x1 = x_c - w / 2
                y1 = y_c - h / 2
                x2 = x_c + w / 2
                y2 = y_c + h / 2

                boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)  # (M,4)

                # Reescalar a tamaño original
                gain_w = orig_w / float(net_w)
                gain_h = orig_h / float(net_h)
                boxes_xyxy[:, [0, 2]] *= gain_w
                boxes_xyxy[:, [1, 3]] *= gain_h

                # Clipping
                boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w - 1)
                boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w - 1)
                boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h - 1)
                boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h - 1)

                # NMS (agnóstico de clase)
                keep = nms(boxes_xyxy, scores_f, iou_thresh=iou_thres)
                boxes_xyxy = boxes_xyxy[keep]
                scores_nms = scores_f[keep]
                class_ids_nms = class_ids_f[keep]
                num_after_nms = len(keep)

            # FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time))
            prev_time = now

            # Dibujar cajas y FPS
            if num_after_nms > 0:
                frame = draw_boxes_multiclass(frame, boxes_xyxy, scores_nms, class_ids_nms, class_names)

            cv2.putText(
                frame,
                f"FPS: {fps:4.1f} Det:{num_after_nms}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Mostrar ventana
            cv2.imshow("Xtion + TensorRT detecciones", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Tecla 'q' pulsada. Saliendo...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por el usuario (Ctrl+C).")

    # Limpieza
    rgb.stop()
    openni2.unload()
    cv2.destroyAllWindows()
    print("[INFO] Finalizado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        type=str,
        default="best.engine",
        help="Ruta al engine TensorRT (.engine)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="classes.txt",        # <<<
        help="Fichero con nombres de clase (uno por línea)",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.05,                 # puedes ajustar según tu modelo
        help="Umbral de confianza",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.5,
        help="Umbral IOU para NMS",
    )

    args = parser.parse_args()
    main(args)
