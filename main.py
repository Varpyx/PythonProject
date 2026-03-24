import cv2

# 1. Načtení trénovaného modelu pro detekci obličejů (součást OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Připojení k webkameře (0 je obvykle výchozí kamera)
cap = cv2.VideoCapture(0)

print("Stiskni 'q' pro ukončení programu.")

while True:
    # Čtení aktuálního snímku z kamery
    ret, frame = cap.read()
    if not ret:
        break

    # Převod do šedotónu (detekce funguje lépe a rychleji bez barev)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Samotná detekce obličejů
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 4. Vykreslení obdélníku kolem každého nalezeného obličeje
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Oblicej', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Zobrazení výsledného okna
    cv2.imshow('Detekce obliceje v realnem case', frame)

    # Ukončení při stisku klávesy 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Vyčištění paměti a zavření oken
cap.release()
cv2.destroyAllWindows()