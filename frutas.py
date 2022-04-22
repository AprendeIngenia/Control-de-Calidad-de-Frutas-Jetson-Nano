import jetson.inference
import jetson.utils as jetu
import numpy as np
import cv2

# Declarar el detector
net = jetson.inference.detectNet(argv=['--model=frutas.onnx', '--labels=labels.txt', '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes'])

# Declaramos la camara y la ventana
camara = jetson.utils.videoSource("/dev/video0")
display = jetson.utils.videoOutput()
ventana = jetson.utils.videoOutput()

# Oxidacion Manzana
cafeb = np.array([10, 150, 50], np.uint8)
cafea = np.array([30, 200, 205], np.uint8)

# Hongo naranja
blancob = np.array([0, 0, 200], np.uint8)
blancoa = np.array([179, 5, 255], np.uint8)

# Manchas banano
negrob = np.array([0, 0, 0], np.uint8)
negroa = np.array([50, 150, 50], np.uint8)



while True:
    # Frames
    img = camara.Capture()

    # Claves
    keym = 0
    keyn = 0
    keyb = 0

    # Imagenes numpy
    frame = jetu.cudaToNumpy(img)

    # Realizamos la deteccion
    detect = net.Detect(img, overlay = 'none')

    # Declaramos listas
    xlista = []
    ylista = []

    # Si hay detecciones
    if detect:
        for det in detect:
            # Determinamos que fruta es
            clase = det.ClassID

            # Si es manzana
            if clase == 1:
                #print("Manzana")

                # Extraemos coordenadas
                xim, yim = det.Left, det.Top
                xfm, yfm = det.Width + xim, det.Top + det.Height
                #jetu.cudaDrawRect(img, (xim, yim, xfm, yfm), (0, 255, 0, 80))

                # Guardamos coordenadas para evitar errores
                xlista.append(xim)
                xlista.append(xfm)
                ylista.append(yim)
                ylista.append(yfm)

                xminm, xmaxm = int(min(xlista)), int(max(xlista))
                yminm, ymaxm = int(min(ylista)), int(max(ylista))

                # Extraemos zona de interes
                recortem = frame[yminm:ymaxm, xminm:xmaxm]

                # Conversion a HSV
                hsvm = cv2.cvtColor(recortem, cv2.COLOR_RGB2HSV)

                # Rango
                maskm = cv2.inRange(hsvm, cafeb, cafea)

                # Contornos
                contornosm, _ = cv2.findContours(maskm, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                # Ordenamos
                contornosm = sorted(contornosm, key=lambda x: cv2.contourArea(x), reverse=True)


                for contm in contornosm:
                    # Extraemos el area
                    aream = cv2.contourArea(contm)

                    if aream >= 50 and aream <= 5000:
                        # Detectamos las zonas malas
                        xsim, ysim, anchom, altom = cv2.boundingRect(contm)

                        # Mostramos los errores
                        jetu.cudaDrawRect(img, (xim + xsim, yim + ysim, xim + xsim + anchom, yim + ysim + altom), (255, 0, 0, 80))

                        # Manzana Mala
                        print("MANZANA EN MAL ESTADO")

                        keym = 1

                        # Convertimos imagen CUDA
                        ven = jetu.cudaFromNumpy(recortem)
                        # Mostramos zona de ineteres
                        ventana.Render(ven)

                if keym == 0:
                    # Manzana buena
                    jetu.cudaDrawRect(img, (xim, yim, xfm, yfm), (0, 255, 0, 80))
                    print("MANZANA EN BUEN ESTADO")



            # Si es banano
            elif clase == 2:
                #print("Banano")

                # Extraemos coordenadas
                xib, yib = det.Left, det.Top
                xfb, yfb = det.Width + xib, det.Top + det.Height
                #jetu.cudaDrawRect(img, (xib, yib, xfb, yfb), (255, 0, 0, 80))

                # Guardamos coordenadas para evitar errores
                xlista.append(xib)
                xlista.append(xfb)
                ylista.append(yib)
                ylista.append(yfb)

                xminb, xmaxb = int(min(xlista)), int(max(xlista))
                yminb, ymaxb = int(min(ylista)), int(max(ylista))

                # Extraemos zona de interes
                recorteb = frame[yminb:ymaxb, xminb:xmaxb]

                # Conversion a HSV
                hsvb = cv2.cvtColor(recorteb, cv2.COLOR_RGB2HSV)

                # Rango
                maskb = cv2.inRange(hsvb, negrob, negroa)

                # Contornos
                contornosb, _ = cv2.findContours(maskb, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                # Ordenamos
                contornosb = sorted(contornosb, key=lambda x: cv2.contourArea(x), reverse=True)

                for contb in contornosb:
                    # Extraemos el area
                    areab = cv2.contourArea(contb)

                    if areab >= 10 and areab <= 50000:
                        # Detectamos las zonas malas
                        xsib, ysib, anchob, altob = cv2.boundingRect(contb)

                        # Mostramos los errores
                        jetu.cudaDrawRect(img, (xib + xsib, yib + ysib, xib + xsib + anchob, yib + ysib + altob),
                                          (255, 0, 0, 80))

                        # Manzana Mala
                        print("BANANO EN MAL ESTADO")

                        keyb = 1

                        # Convertimos imagen CUDA
                        ven = jetu.cudaFromNumpy(recorteb)
                        # Mostramos zona de ineteres
                        ventana.Render(ven)

                if keyb == 0:
                    # Banano buena
                    jetu.cudaDrawRect(img, (xib, yib, xfb, yfb), (0, 255, 0, 80))
                    print("BANANO EN BUEN ESTADO")

            elif clase == 3:
                #print("Naranja")

                # Extraemos coordenadas
                xin, yin = det.Left, det.Top
                xfn, yfn = det.Width + xin, det.Top + det.Height
                #jetu.cudaDrawRect(img, (xin, yin, xfn, yfn), (0, 0, 255, 80))

                # Guardamos coordenadas para evitar errores
                xlista.append(xin)
                xlista.append(xfn)
                ylista.append(yin)
                ylista.append(yfn)

                xminn, xmaxn = int(min(xlista)), int(max(xlista))
                yminn, ymaxn = int(min(ylista)), int(max(ylista))

                # Extraemos zona de interes
                recorten = frame[yminn:ymaxn, xminn:xmaxn]

                # Conversion a HSV
                hsvn = cv2.cvtColor(recorten, cv2.COLOR_RGB2HSV)

                # Rango
                maskn = cv2.inRange(hsvn, blancob, blancoa)

                # Contornos
                contornosn, _ = cv2.findContours(maskn, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                # Ordenamos
                contornosn = sorted(contornosn, key=lambda x: cv2.contourArea(x), reverse=True)

                for contn in contornosn:
                    # Extraemos el area
                    arean = cv2.contourArea(contn)

                    if arean >= 10 and arean <= 5000:
                        # Detectamos las zonas malas
                        xsin, ysin, anchon, alton = cv2.boundingRect(contn)

                        # Mostramos los errores
                        jetu.cudaDrawRect(img, (xin + xsin, yin + ysin, xin + xsin + anchon, yin + ysin + alton),
                                          (255, 0, 0, 80))

                        # Manzana Mala
                        print("NARANJA EN MAL ESTADO")

                        keyn = 1

                        # Convertimos imagen CUDA
                        #ven = jetu.cudaFromNumpy(recorten)
                        # Mostramos zona de ineteres
                        #ventana.Render(ven)

                if keyn == 0:
                    # Manzana buena
                    jetu.cudaDrawRect(img, (xin, yin, xfn, yfn), (0, 0, 255, 80))
                    print("NARANJA EN BUEN ESTADO")




    # Renderizamos la imagen
    display.Render(img)
    display.SetStatus("FRUTAS | Network {:.0f}FPS".format(net.GetNetworkFPS()))

    # Para cerrar
    if not camara.IsStreaming() or not display.IsStreaming():
        break