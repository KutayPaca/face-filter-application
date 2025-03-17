# Yüz Algılama ve Filtre Uygulama Projesi

Bu proje, OpenCV ve MediaPipe kullanarak bir web kamerası aracılığıyla yüz algılama ve çeşitli filtreler uygulama işlevselliği sağlar.
Kullanıcılar, webcam görüntüsü üzerinde yüzlerini tespit edip, farklı filtreler uygulayabilirler.

## Özellikler:
- **Yüz Algılama**: MediaPipe ile gerçek zamanlı olarak yüz algılaması yapılır.
- **Filtreler**: Yüz bölgesine çeşitli filtreler uygulanabilir. Mevcut filtreler:
  1. Ortalama Filtre
  2. Median Filtre
  3. Gauss Filtre
  4. Sobel Filtre
  5. Prewitt Filtre
  6. Laplacian Filtre
  7. Yüzü Bulanıklaştırma
- **Webcam Entegrasyonu**: Web kamerasından gelen video akışı üzerinden yüz algılaması yapılır.
- **Etkin Kullanıcı Arayüzü**: Klavye ile filtreler arasında geçiş yapılabilir.

## Kullanım

- **Filtre Seçimi**: Filtreler arasında geçiş yapmak için aşağıdaki tuşları kullanabilirsiniz:
  - `1`: Ortalama filtre
  - `2`: Median filtre
  - `3`: Gauss filtresi
  - `4`: Sobel filtresi
  - `5`: Prewitt filtresi
  - `6`: Laplacian filtresi
  - `7`: Yüzü bulanıklaştır
- **Çıkış**: `q` tuşuna basarak uygulamadan çıkabilirsiniz.
