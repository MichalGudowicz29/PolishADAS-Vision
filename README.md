# PolishADAS-Vision

![Project Banner](docs/images/project_banner.jpg)

**Zaawansowany system detekcji znaków drogowych w czasie rzeczywistym**

*System komputerowego widzenia wykrywający i klasyfikujący polskie znaki drogowe oraz linie pasa ruchu w czasie rzeczywistym.*

[![Demo Video](docs/images/video_thumbnail.jpg)](https://youtu.be/your_video_id_here)
*Kliknij powyższy obraz, aby obejrzeć film demonstracyjny*

##  O projekcie

PolishADAS-Vision to kompletny system wizyjny bazujący na głębokim uczeniu, który realizuje podstawowe funkcje systemów ADAS (Advanced Driver Assistance Systems). Projekt wykorzystuje Raspberry Pi Zero 2 W z modułem kamery do przechwytywania obrazu oraz tablet HP Pavilion X2 do przetwarzania i analizy.

System w czasie rzeczywistym:
- Wykrywa i klasyfikuje polskie znaki drogowe (ponad 40 kategorii)
- Identyfikuje linie pasa ruchu
- Filtruje znaki po prawej stronie drogi (istotne dla kierowcy)
- Implementuje wygładzanie temporalne dla stabilnych detekcji

Ten projekt stanowi wartościową demonstrację umiejętności w dziedzinie systemów ADAS, computer vision oraz deep learning - kluczowych technologii w rozwoju nowoczesnych systemów wspomagania kierowcy i pojazdów autonomicznych.

##  Zdjęcia systemu

### Pełny montaż sprzętowy
![System Setup](docs/images/hardware_setup.jpg)
*Montaż systemu z tabletem HP Pavilion X2 oraz modułem kamery na Raspberry Pi Zero 2 W*

### Detekcja znaków w akcji
![Sign Detection](docs/images/sign_detection.jpg)
*System wykrywający i klasyfikujący znaki ograniczenia prędkości*

### Detekcja linii pasa ruchu
![Lane Detection](docs/images/lane_detection.jpg)
*System wykrywający linie pasa ruchu na drodze*

##  Architektura systemu

![System Architecture](docs/images/system_architecture.jpg)

System składa się z dwóch głównych komponentów sprzętowych:

1. **Moduł akwizycji obrazu**
   - Raspberry Pi Zero 2 W
   - Kamera Raspberry Pi Camera V2 (8Mpx)
   - Streaming wideo przez RTSP

2. **Moduł przetwarzania i analizy**
   - Tablet HP Pavilion X2 (część odłączana)
   - YOLOv8 do detekcji znaków
   - MobileNetV2 do klasyfikacji typów znaków
   - Algorytmy detekcji linii pasa ruchu
   - Wielowątkowa architektura dla płynnej pracy

##  Kluczowe funkcje

- **Detekcja i klasyfikacja znaków drogowych** - YOLOv8 + klasyfikator ConvNet
- **Detekcja linii pasa ruchu** - algorytm bazujący na transformacie Hougha
- **Filtrowanie kontekstowe** - analiza tylko znaków po prawej stronie drogi
- **Wygładzanie temporalne** - eliminacja niestabilnych detekcji
- **Wielowątkowe przetwarzanie** - separacja akwizycji i analizy obrazu
- **Interfejs użytkownika** - wyświetlanie informacji o detekcjach w czasie rzeczywistym
- **RTSP streaming** - bezprzewodowa transmisja obrazu z kamery do tabletu

## 🔍 Szczegóły techniczne

### Modele Deep Learning

- **Detekcja**: YOLOv8n (nanio) - lekki model zoptymalizowany pod kątem urządzeń mobilnych
- **Klasyfikacja**: MobileNetV2 dostrojony na zbiorze polskich znaków drogowych
- **Rozdzielczość przetwarzania**: 640×480 pikseli przy 15 FPS
- **Rozdzielczość wyświetlania**: 800×600 pikseli

### Przetwarzanie obrazu

- Dwuetapowy pipeline detekcji i klasyfikacji
- Detekcja krawędzi Canny'ego do identyfikacji linii pasa ruchu
- Transformata Hougha do konwersji punktów krawędzi na linie
- Region zainteresowania (ROI) dla wydzielenia obszaru drogi
- Wygładzanie temporalne z pamięcią 5 ostatnich klatek

### Zbiór danych

Projekt wykorzystuje zbiór polskich znaków drogowych zawierający:
- 40+ kategorii znaków
- Dane do detekcji (pełne obrazy z bounding boxami)
- Dane do klasyfikacji (wydzielone obrazy znaków poszczególnych kategorii)

##  Instalacja i użytkowanie

### Wymagania

#### Sprzętowe
- Raspberry Pi Zero 2 W
- Raspberry Pi Camera Module V2
- Kabel FPC adapter (15-pin do 22-pin)
- Tablet/laptop do przetwarzania
- Zasilanie dla Raspberry Pi

#### Programowe
- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLOv8
- NumPy
- PIL/Pillow

### Konfiguracja Raspberry Pi

```bash
# Po połączeniu z Raspberry Pi przez SSH
sudo python3 pi_setup.py --wifi-ssid "NazwaWiFi" --wifi-password "HasłoWiFi"
sudo reboot

# Po restarcie, sprawdź działanie streamu
python3 pi_setup.py --test
```

### Uruchomienie systemu

```bash
# Na tablecie/laptopie
python main.py --source "rtsp://192.168.1.200:8554/cam" --detection_model best.pt --classification_model sign_classifier.pth --detect_lanes
```

##  Zastosowania

System może być wykorzystany jako:
- Prototyp systemu ADAS do wczesnych testów
- Platforma edukacyjna do nauczania computer vision
- Narzędzie do zbierania i walidacji danych w rzeczywistych warunkach drogowych
- Podstawa do rozbudowy o bardziej zaawansowane funkcje ADAS

##  Przyszły rozwój

Planowane rozszerzenia projektu:
- Integracja z OBD-II dla dostępu do danych pojazdu
- Detekcja i śledzenie innych uczestników ruchu drogowego
- Implementacja estymacji odległości (monocular depth estimation)
- Optymalizacja wydajności przez kwantyzację modeli
- Rozbudowa interfejsu użytkownika o dodatkowe funkcje

##  Zasoby i dokumentacja

- [Szczegółowa dokumentacja architektury](docs/architecture.md)
- [Instrukcja trenowania modeli](docs/training.md)
- [Specyfikacja API](docs/api.md)
- [Nota metodologiczna](docs/methodology.md)

##  Podziękowania

- [Ultralytics](https://github.com/ultralytics/ultralytics) za framework YOLOv8
- [PyTorch](https://pytorch.org/) za bibliotekę deep learning
- [OpenCV](https://opencv.org/) za narzędzia do przetwarzania obrazu
- [rtsp-simple-server](https://github.com/aler9/rtsp-simple-server) za lekki serwer RTSP

##  Licencja

Ten projekt jest udostępniony na licencji MIT. Szczegóły w pliku [LICENSE](LICENSE).

---

*Ten projekt został stworzony jako część portfolio dla branży ADAS/pojazdów autonomicznych. Nie jest przeznaczony do użytku komercyjnego ani do użytku w prawdziwych pojazdach bez odpowiednich certyfikacji bezpieczeństwa.*
