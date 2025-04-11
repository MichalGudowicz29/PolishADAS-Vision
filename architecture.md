# Architektura Systemu Detekcji Znaków Drogowych

## 1. Wprowadzenie do architektury systemu

System detekcji znaków drogowych stanowi prototypową implementację technologii ADAS (Advanced Driver Assistance Systems), wykorzystującą komputerową analizę obrazu do wykrywania, klasyfikacji i interpretacji znaków drogowych w czasie rzeczywistym. Architektura systemu odzwierciedla typowe rozwiązania stosowane w komercyjnych systemach ADAS, z uwzględnieniem ograniczeń sprzętowych platformy docelowej (tablet HP Pavilion X2 oraz Raspberry Pi Zero 2 W).

## 2. Architektura sprzętowa

System został zbudowany w oparciu o dwuczęściową architekturę sprzętową:

1. **Moduł akwizycji obrazu:**
   - Raspberry Pi Zero 2 W z modułem kamery V2 (8 Mpx)
   - System operacyjny: Raspberry Pi OS Lite
   - Transmisja danych: strumieniowanie RTSP przez sieć WiFi
   - Rozdzielczość przechwytywania: 640×480 pikseli
   - Częstotliwość odświeżania: 15 klatek na sekundę

2. **Moduł przetwarzania i wyświetlania:**
   - Część tabletowa HP Pavilion Detachable X2
   - System operacyjny: Windows/Linux
   - Funkcje: przetwarzanie obrazu, detekcja znaków, klasyfikacja, wizualizacja wyników

Taka dwuczęściowa architektura ma kluczowe zalety w systemach ADAS:
- Rozdzielenie akwizycji od przetwarzania pozwala na optymalne umiejscowienie kamery
- Zmniejsza obciążenie cieplne głównego procesora
- Umożliwia redundancję w przypadku awarii jednego z podsystemów
- Pozwala na modułową rozbudowę systemu

## 3. Przetwarzanie strumienia wideo - architektura programowa

### 3.1. Wielowątkowy model przetwarzania

```python
def run(self):
    """Main method to run the system"""
    self.running = True
    
    # Start processing thread
    processing_thread = threading.Thread(target=self.process_frames)
    processing_thread.daemon = True
    processing_thread.start()
    
    try:
        while self.running:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to receive frame. Exiting...")
                break
            
            # Store the frame for processing
            self.frame = frame
            
            # Display the processed frame if available
            if self.processed_frame is not None:
                cv2.imshow("Polish Traffic Sign Detection", self.processed_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                # Toggle lane detection
                self.args.detect_lanes = not self.args.detect_lanes
```

System implementuje wielowątkową architekturę przetwarzania, która jest kluczowa w systemach ADAS z następujących powodów:

1. **Separacja akwizycji i przetwarzania**: Główny wątek zajmuje się wyłącznie pobieraniem klatek ze strumienia RTSP, minimalizując ryzyko opóźnień i utraty danych.

2. **Zapobieganie "jitterowi"**: Separacja wątków pozwala na stabilne wyświetlanie przetworzonych klatek nawet gdy czas obliczeń dla poszczególnych klatek jest zmienny.

3. **Responsywność UI**: Interfejs użytkownika pozostaje responsywny nawet podczas intensywnego przetwarzania.

W rzeczywistych systemach ADAS ta architektura jest rozszerzana często do modelu producent-konsument z buforowaniem klatek.

### 3.2. Pobieranie i przygotowanie strumienia wideo

```python
def setup_video_capture(self):
    try:
        if self.source.startswith('rtsp://'):
            # For RTSP stream from Raspberry Pi
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce latency
        elif self.source.isdigit():
            # For webcam
            self.cap = cv2.VideoCapture(int(self.source))
        else:
            # For video file
            self.cap = cv2.VideoCapture(self.source)
            
        # Check if camera/video opened successfully
        if not self.cap.isOpened():
            print(f"Error: Unable to open video source {self.source}")
            exit(1)
            
        # Set resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESSING_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESSING_HEIGHT)
```

Ta funkcja obsługuje inicjalizację źródła wideo z kluczowymi optymalizacjami dla systemów ADAS:

1. **Elastyczny wybór źródła**: Umożliwia testowanie z różnymi źródłami (kamera, plik, strumień), co ma kluczowe znaczenie w procesie rozwoju systemów ADAS.

2. **Specyficzna konfiguracja dla RTSP**: Wykorzystanie backendu FFMPEG i minimalnego bufora (2 klatki) maksymalizuje aktualność przetwarzanych danych, co jest krytyczne w systemach bezpieczeństwa.

3. **Kontrola rozdzielczości**: Świadoma redukcja rozdzielczości do 640×480 to kompromis między dokładnością detekcji a wydajnością przetwarzania na sprzęcie mobilnym.

W profesjonalnych systemach ADAS podobne strategie są stosowane do balansu między opóźnieniem przetwarzania a wymaganiami obliczeniowymi.

## 4. Architektura przetwarzania i detekcji

### 4.1. Dwuetapowy proces detekcji-klasyfikacji

```python
def detect_signs(self, frame):
    try:
        # Run YOLOv8 detection
        results = self.detection_model(frame, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                
                # Right-side filtering (only consider signs on the right side of frame)
                center_x = (x1 + x2) / 2
                if center_x > frame.shape[1] / 2:
                    # Extract the sign region for classification
                    sign_img = frame[int(y1):int(y2), int(x1):int(x2)]
                    
                    if sign_img.size == 0:  # Skip empty detections
                        continue
                        
                    # Convert to PIL for classification preprocessing
                    sign_img_pil = Image.fromarray(cv2.cvtColor(sign_img, cv2.COLOR_BGR2RGB))
                    input_tensor = self.preprocess(sign_img_pil).unsqueeze(0)
                    
                    # Classify the sign
                    with torch.no_grad():
                        output = self.classifier(input_tensor)
                        _, predicted_idx = torch.max(output, 1)
                        sign_class = predicted_idx.item()
                        sign_name = self.class_names[sign_class]
                    
                    detections.append({
                        'class': sign_class,
                        'name': sign_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2)
                    })
```

Architektura dwuetapowa detekcja-klasyfikacja jest kluczowym podejściem w systemach ADAS z następujących powodów:

1. **Specjalizacja modeli**: Model YOLOv8 jest zoptymalizowany do lokalizacji obiektów (gdzie są znaki), a oddzielny klasyfikator ConvNet do rozpoznawania typów znaków (jakie to znaki).

2. **Filtrowanie kontekstowe**: Implementacja filtrowania po prawej stronie kadru wykorzystuje wiedzę dziedzinową (znaki drogowe istotne dla kierowcy znajdują się po prawej stronie drogi w Polsce), co jest powszechną praktyką w systemach ADAS.

3. **Progowanie pewności**: Eliminacja detekcji o niskiej pewności (< CONFIDENCE_THRESHOLD) zapobiega fałszywym alarmom.

4. **Izolacja regionów zainteresowania (ROI)**: Wycinanie i przetwarzanie tylko interesujących obszarów redukuje obciążenie obliczeniowe.

5. **Redukcja wymiarowości**: Konwersja współrzędnych z formatu YOLO do współrzędnych obrazu demonstruje typową praktykę przetwarzania danych w pipeline'ach computer vision.

### 4.2. Detekcja linii pasa ruchu

```python
def detect_lanes(self, frame):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # Define region of interest (bottom half of the image)
        height, width = edges.shape
        mask = np.zeros_like(edges)
        # Define a polygon for the ROI - focus on the road area
        polygon = np.array([
            [(0, height), (width, height), (width//2, height//2), (width//2, height//2), (0, height)]
        ], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 50, 
                               minLineLength=40, maxLineGap=5)
        
        # Create blend image for lane lines
        line_image = np.zeros_like(frame)
        
        # Variables to store lane line parameters
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate slope to separate left and right lanes
                if x2 - x1 == 0:  # Avoid division by zero
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter out horizontal lines
                if abs(slope) < 0.5:
                    continue
                    
                # Separate left and right lanes based on slope
                if slope < 0:  # Left lane
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:  # Right lane
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])
```

Detekcja linii pasa ruchu implementuje klasyczny algorytm wykorzystywany w systemach ADAS, prezentując kluczowe etapy:

1. **Preprocessing**: Konwersja do skali szarości i rozmycie gaussowskie redukują szum i normalizują dane wejściowe.

2. **Detekcja krawędzi**: Algorytm Canny'ego identyfikuje krawędzie, co jest podstawowym krokiem w identyfikacji elementów strukturalnych.

3. **Region zainteresowania (ROI)**: Definicja trapezoidalnego obszaru zainteresowania to kluczowa optymalizacja w systemach detekcji pasów ruchu, eliminująca irrelewantne części obrazu (niebo, pobocze).

4. **Transformata Hougha**: Algorytm przekształca punkty krawędzi w potencjalne linie, co jest standardową techniką w detekcji elementów liniowych.

5. **Klasyfikacja linii**: Podział na lewe i prawe linie pasa na podstawie nachylenia wykorzystuje wiedzę geometryczną o układzie dróg.

6. **Filtrowanie linii poziomych**: Eliminacja linii o małym nachyleniu (<0.5) zapobiega wykrywaniu krawężników, progów i innych elementów poziomych jako pasów ruchu.

Te techniki stanowią podstawę systemów detekcji pasów ruchu w komercyjnych rozwiązaniach ADAS, choć zaawansowane systemy implementują dodatkowo śledzenie czasowe i modele predykcyjne.

### 4.3. Wygładzanie temporalne detekcji

```python
def temporal_smoothing(self):
    """Smooth detections over multiple frames to reduce flickering"""
    if not self.sign_history:
        return []
        
    # Count occurrences of each sign class
    sign_counts = {}
    for frame_detections in self.sign_history:
        for det in frame_detections:
            sign_name = det['name']
            if sign_name not in sign_counts:
                sign_counts[sign_name] = {
                    'count': 1,
                    'confidence': det['confidence'],
                    'bbox': det['bbox'],
                    'class': det['class']
                }
            else:
                sign_counts[sign_name]['count'] += 1
                # Update with highest confidence detection
                if det['confidence'] > sign_counts[sign_name]['confidence']:
                    sign_counts[sign_name]['confidence'] = det['confidence']
                    sign_counts[sign_name]['bbox'] = det['bbox']
    
    # Only keep signs that appear in multiple frames
    min_count = max(1, len(self.sign_history) // 3)
    smoothed = []
    for sign_name, data in sign_counts.items():
        if data['count'] >= min_count:
            smoothed.append({
                'name': sign_name,
                'confidence': data['confidence'],
                'bbox': data['bbox'],
                'class': data['class']
            })
```

Wygładzanie temporalne to krytyczny element w systemach ADAS, który zapewnia stabilność detekcji:

1. **Redukcja drgań (jitter)**: Algorytm eliminuje krótkotrwałe, pojedyncze detekcje, które mogą być artefaktami lub błędami.

2. **Pamięć historyczna**: System utrzymuje historię detekcji z kilku ostatnich klatek (KEEP_FRAMES = 5), co implementuje prostą formę śledzenia obiektów.

3. **Głosowanie większościowe**: Znaki muszą być obecne w co najmniej 1/3 zachowanych klatek, aby zostały uznane za prawidłowe detekcje.

4. **Selekcja najlepszej instancji**: Z wielu detekcji tego samego znaku system wybiera tę o najwyższej pewności.

Ten mechanizm naśladuje bardziej zaawansowane systemy filtrów (np. filtr Kalmana) stosowane w komercyjnych ADAS, a jednocześnie pozostaje obliczeniowo efektywny.

## 5. Architektura modeli uczenia maszynowego

### 5.1. Model detekcji YOLOv8

```python
# Load detection model (YOLOv8 for sign detection)
try:
    from ultralytics import YOLO
    self.detection_model = YOLO(args.detection_model)
    print(f"Loaded detection model: {args.detection_model}")
except Exception as e:
    print(f"Error loading detection model: {e}")
    exit(1)
```

Model YOLOv8 stanowi fundament systemu detekcji z kilku kluczowych powodów:

1. **Architektura jednoetapowa**: W odróżnieniu od dwuetapowych modeli (jak R-CNN), YOLO wykonuje detekcję i lokalizację w jednym przebiegu sieci, co jest kluczowe dla aplikacji czasu rzeczywistego.

2. **Zrównoważenie szybkości i dokładności**: YOLOv8n zapewnia kompromis między precyzją a wymaganiami obliczeniowymi, co jest krytyczne dla urządzeń z ograniczoną mocą obliczeniową.

3. **Elastyczność wdrożenia**: Biblioteka Ultralytics umożliwia łatwe wdrożenie na różnych platformach, w tym urządzeniach mobilnych.

W systemach ADAS, wybór odpowiedniego detektora jest krytyczny i często preferowane są modele jednoetapowe ze względu na niższe opóźnienia, nawet kosztem niewielkiego spadku dokładności.

### 5.2. Model klasyfikacji znaków

```python
class SignClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SignClassifier, self).__init__()
        # Use MobileNetV2 for better performance on the tablet
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        
    def forward(self, x):
        return self.model(x)
```

Klasyfikator znaków bazuje na architekturze MobileNetV2, co stanowi strategiczny wybór dla systemów ADAS działających na urządzeniach mobilnych:

1. **Efektywność obliczeniowa**: MobileNetV2 wykorzystuje separowalne konwolucje głębokie (depthwise separable convolutions), które drastycznie redukują liczbę operacji w porównaniu do standardowych CNN.

2. **Transfer learning**: Wykorzystanie modelu wstępnie wytrenowanego na ImageNet przyspiesza konwergencję i poprawia generalizację na małych zbiorach danych.

3. **Adaptacja do zadania**: Zastąpienie końcowej warstwy klasyfikacyjnej dostosowuje model do specyficznego zestawu klas znaków drogowych.

W komercyjnych systemach ADAS podobne podejście z lekką architekturą sieci jest powszechne, choć często stosuje się również kwantyzację modeli (int8) dla dalszej optymalizacji.

### 5.3. Przygotowanie danych i trenowanie modeli

```python
def train_classifier(args):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transformations for training with augmentations
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
```

Proces trenowania klasyfikatora implementuje kluczowe praktyki rozwoju modeli dla systemów ADAS:

1. **Augmentacja danych**: Transformacje jak obroty, odbicia i zmiany jasności zwiększają różnorodność danych treningowych, zapewniając odporność na zmienne warunki środowiskowe.

2. **Normalizacja danych**: Standaryzacja do specyficznych średnich i odchyleń standardowych (wartości ImageNet) optymalizuje działanie modelu wstępnie trenowanego.

3. **Adaptacja sprzętowa**: Automatyczne wykrywanie dostępności GPU pozwala na efektywne wykorzystanie dostępnych zasobów.

4. **Rozdzielczość wejściowa**: Standaryzacja rozmiaru (224×224) balansuje szczegółowość z wymaganiami pamięciowymi.

Te techniki są szeroko stosowane w przemysłowych procesach trenowania modeli dla systemów ADAS.

## 6. Implementacja na Raspberry Pi

```python
def setup_rtsp_server():
    """Install and configure rtsp-simple-server"""
    print("\n=== Setting up RTSP Server ===")
    
    # Install required packages
    run_command("sudo apt update")
    run_command("sudo apt install -y git wget libcamera-apps-lite")
    
    # Download rtsp-simple-server
    run_command("cd /tmp && wget https://github.com/aler9/rtsp-simple-server/releases/download/v0.21.0/rtsp-simple-server_v0.21.0_linux_armv6.tar.gz")
    run_command("cd /tmp && tar -xzf rtsp-simple-server_v0.21.0_linux_armv6.tar.gz")
    run_command("sudo mv /tmp/rtsp-simple-server /usr/local/bin/")
```

Implementacja na Raspberry Pi demonstruje typowe podejście do systemów akwizycji obrazu w ADAS:

1. **Lekki serwer RTSP**: Wykorzystanie rtsp-simple-server minimalizuje obciążenie procesora Raspberry Pi, pozostawiając zasoby dla przechwytywania obrazu.

2. **Niskoopoźnieniowe przesyłanie**: Konfiguracja bezpośredniego strumieniowania TCP minimalizuje opóźnienia, co jest krytyczne w systemach bezpieczeństwa.

3. **Autostart usług**: Wykorzystanie systemd do automatycznego uruchamiania usług po starcie systemu zapewnia niezawodne działanie bez ingerencji użytkownika.

4. **Dedykowana konfiguracja sieciowa**: Statyczne przypisanie adresu IP (192.168.1.200) zapewnia stabilne połączenie między modułami systemu.

Te praktyki odzwierciedlają podejście stosowane w przemysłowych systemach ADAS, gdzie moduły kamer muszą działać niezawodnie w różnych warunkach.

## 7. Interfejs użytkownika i wizualizacja

```python
def draw_ui(self, frame, detections):
    """Draw the user interface with detections and information"""
    try:
        # Resize frame for display
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = [int(coord * DISPLAY_WIDTH / PROCESSING_WIDTH) 
                             if i % 2 == 0 else int(coord * DISPLAY_HEIGHT / PROCESSING_HEIGHT) 
                             for i, coord in enumerate(det['bbox'])]
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with confidence
            label = f"{det['name']} ({det['confidence']:.2f})"
            cv2.rectangle(display_frame, (x1, y1-25), (x1+len(label)*8, y1), (0, 255, 0), -1)
            cv2.putText(display_frame, label, (x1, y1-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
```

Interfejs użytkownika implementuje praktyki wizualizacji typowe dla prototypowych systemów ADAS:

1. **Rozdzielenie procesowania i wyświetlania**: Kod przelicza współrzędne detekcji z rozdzielczości przetwarzania (640×480) na rozdzielczość wyświetlania (800×600).

2. **Informacyjne etykiety**: Wyświetlanie nazwy znaku i pewności detekcji dostarcza kluczowych informacji diagnostycznych.

3. **Wskaźniki wydajności**: Licznik FPS pozwala na bieżąco monitorować wydajność systemu.

4. **Interaktywne sterowanie**: Obsługa przełączania funkcji (np. detekcji linii) przez naciśnięcie klawiszy zapewnia elastyczność testowania.

W zaawansowanych systemach ADAS, interfejs użytkownika jest często bardziej zminimalizowany lub zintegrowany z systemem inforozrywki pojazdu.

## 8. Wnioski i potencjał rozwojowy

Zaprezentowany system detekcji znaków drogowych stanowi fundament dla bardziej zaawansowanych systemów ADAS. Architektura systemu umożliwia:

1. **Skalowalność**: Dodawanie nowych klas znaków drogowych bez zmian w kodzie bazowym.

2. **Rozszerzalność**: Integrację z dodatkowymi modułami ADAS, takimi jak detekcja przeszkód czy monitorowanie uwagi kierowcy.

3. **Optymalizację wydajności**: Implementację technik takich jak kwantyzacja modeli czy przetwarzanie na poziomie tensorów.

4. **Poprawę dokładności**: Zastosowanie bardziej zaawansowanych technik śledzenia obiektów i filtrowania temporalnego.

W kontekście przemysłowych systemów ADAS, przedstawiona architektura stanowi solidny punkt wyjścia, wymagający dalszej optymalizacji pod kątem niezawodności, certyfikacji bezpieczeństwa i integracji z systemami kontroli pojazdu.

## 9. Bibliografia

1. Redmon, J., & Farhadi, A. (2018). YOLOv3: An incremental improvement. arXiv preprint arXiv:1804.02767.
2. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861.
3. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.
5. Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp, B., Goyal, P., ... & Zhang, X. (2016). End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316.
```