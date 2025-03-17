import cv2
import mediapipe as mp
import numpy as np

# Görselleştirme için sabitler
COLOR = (255, 0, 0)  # Mavi renk (BGR formatında)

# MediaPipe Yüz Algılama modülünü başlat
mp_face_detection = mp.solutions.face_detection

# Çerçeve işleme fonksiyonu: Yüzleri algılar ve işlenmiş çerçeveyi ve koordinatları döner
def process_frame(image, draw_box=True):
    """
    Bu fonksiyon, bir görüntüde yüz algılaması yapar ve yüzlerin etrafına sınırlayıcı kutular çizer.
    
    Parametreler:
        - image: İşlenecek görüntü (BGR formatında bir OpenCV çerçevesi).
        - draw_box (bool, opsiyonel): Yüzlerin etrafına sınırlayıcı kutular çizilip çizilmeyeceğini belirler.
          Varsayılan olarak True (kutular çizilir).
    
    Dönüş Değerleri:
        - annotated_image: Yüzlerin etrafına kutular çizilmiş (veya çizilmemiş) işlenmiş görüntü.
        - coordinates: Algılanan her yüzün (origin_x, origin_y, bbox_width, bbox_height) formatında
          koordinatlarını içeren bir liste.
    """
    
    # MediaPipe Yüz Algılama nesnesini başlat
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Çerçevenin kopyasını oluştur (orijinal çerçeveyi değiştirmemek için)
    annotated_image = image.copy()
    height, width, _ = image.shape  # Çerçevenin yüksekliğini, genişliğini ve kanal sayısını al
    coordinates = []  # Yüz koordinatlarını saklamak için liste

    # Yüz algılama işlemini gerçekleştir (RGB formatında)
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Yüzler algılandıysa kontrol et
    if results.detections:
        for detection in results.detections:
            # sınırlayıcı kutunun (bounding box) bilgilerini al
            bboxC = detection.location_data.relative_bounding_box
            origin_x = int(bboxC.xmin * width)  # Sol üst köşe x koordinatı
            origin_y = int(bboxC.ymin * height)  # Sol üst köşe y koordinatı
            bbox_width = int(bboxC.width * width)  # Kutunun genişliği
            bbox_height = int(bboxC.height * height)  # Kutunun yüksekliği

            # Koordinatları sınırların içinde kalacak şekilde ayarla
            origin_x = max(0, min(origin_x, width - 1))
            origin_y = max(0, min(origin_y, height - 1))
            bbox_width = max(0, min(bbox_width, width - origin_x))
            bbox_height = max(0, min(bbox_height, height - origin_y))

            # Koordinatları listeye ekle
            coordinates.append((origin_x, origin_y, bbox_width, bbox_height))

            # Eğer draw_box True ise sınırlayıcı kutuyu çiz
            if draw_box:
                start_point = (origin_x, origin_y)  # sınırlayıcı kutunun başlangıç noktası
                end_point = (origin_x + bbox_width, origin_y + bbox_height)  # sınırlayıcı kutunun bitiş noktası
                cv2.rectangle(annotated_image, start_point, end_point, COLOR, 3)  # Mavi kutu çiz

    # İşlenmiş çerçeveyi ve koordinatları döndür
    return annotated_image, coordinates

# Yüz bölgesine filtre uygulama fonksiyonu
def apply_filter(frame, face_coordinates, filter_type):
    """
    Yüz bölgesine seçilen filtreyi uygular.
    
    Parametreler:
        - frame: Orijinal görüntü.
        - face_coordinates: Yüzün koordinatlarını içeren (x, y, width, height) formatındaki liste.
        - filter_type: Uygulanacak filtre tipi (int, 1 ile 7 arasında).
    """
    for (x, y, w, h) in face_coordinates:
        face_roi = frame[y:y+h, x:x+w]  # Yüz bölgesini al

        # Filtre tipine göre işlemi uygula
        if filter_type == 1:
            filtered_face = cv2.blur(face_roi, (15, 15))  # Ortalama filtre
        elif filter_type == 2:
            filtered_face = cv2.medianBlur(face_roi, 15)  # Median filtre
        elif filter_type == 3:
            filtered_face = cv2.GaussianBlur(face_roi, (15, 15), 0)  # Gauss filtresi
        elif filter_type == 4:
            sobelx = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=5)
            filtered_face = cv2.magnitude(sobelx, sobely)  # Sobel filtresi
        elif filter_type == 5:
            kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            filtered_face = cv2.filter2D(face_roi, -1, kernelx + kernely)  # Prewitt filtresi
        elif filter_type == 6:
            filtered_face = cv2.Laplacian(face_roi, cv2.CV_64F)  # Laplacian filtresi
        elif filter_type == 7:
            filtered_face = cv2.GaussianBlur(face_roi, (99, 99), 30)  # Yüzü bulanıklaştır

        # Filtrelenmiş yüzü orijinal çerçeveye geri yerleştir
        frame[y:y+h, x:x+w] = cv2.convertScaleAbs(filtered_face)

# Webcam'den görüntü al ve yüz algılama uygula
cap = cv2.VideoCapture(0)  # Webcami başlat

current_filter = None  # Şu anki filtre türü (başlangıçta yok)

while cap.isOpened():  # Webcam açık olduğu sürece
    ret, frame = cap.read()  # Bir çerçeve oku
    if not ret:  # Eğer çerçeve okunamadıysa
        print("Hata: Çerçeve yakalanamadı.")
        break

    # process_frame fonksiyonunu çağır, draw_box parametresiyle
    annotated_frame, face_coordinates = process_frame(frame, draw_box=True)

    # Eğer bir filtre seçiliyse, yüz bölgesine uygula
    if current_filter:
        apply_filter(annotated_frame, face_coordinates, current_filter)

    # İşlenmiş çerçeveyi göster
    cv2.imshow("Webcam", annotated_frame)

    # Klavye girişini kontrol et
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):  # 'q' tuşuna basıldığında döngüyü kır (çık)
        break
    elif key == ord("1"):
        current_filter = 1  # Ortalama filtre
    elif key == ord("2"):
        current_filter = 2  # Median filtre
    elif key == ord("3"):
        current_filter = 3  # Gauss filtresi
    elif key == ord("4"):
        current_filter = 4  # Sobel filtresi
    elif key == ord("5"):
        current_filter = 5  # Prewitt filtresi
    elif key == ord("6"):
        current_filter = 6  # Laplacian filtresi
    elif key == ord("7"):
        current_filter = 7  # Yüzü bulanıklaştır

# Webcami serbest bırak ve tüm OpenCV pencerelerini kapat
cap.release()
cv2.destroyAllWindows()
