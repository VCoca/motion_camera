# Sistem za detekciju prisustva osoba

Dvostepeni sistem na platformi Raspberry Pi 4 Model B. Prvi stepen je pasivni
infracrveni senzor HC-SR501 koji neprekidno nadgleda prostor; njegovo reagovanje
pokreće drugi stepen, u kome USB kamera daje sliku, a postupak otkrivanja osoba
utvrđuje da li je reč o čoveku. Stanje se prikazuje svetlećom diodom i veb
stranom koju server na ploči sastavlja na zahtev.

Praktični deo završnog rada; teorijske osnove su obrađene u istraživačkom radu
koji mu prethodi.

## Povezivanje

| HC-SR501 | Raspberry Pi 4B |
|----------|-----------------|
| VCC      | 5 V (pin 2)     |
| OUT      | GPIO 17 (pin 11) |
| GND      | GND (pin 6)     |

Izlaz modula je nivoa 3,3 V, pa je saglasan sa ulazima konektora i pretvarač
logičkih nivoa nije potreban. Kratkospojnik na modulu u položaj **H**
(ponovljivo okidanje).

Indikatorska dioda: GPIO 18 (pin 12) → otpornik 270 Ω → dioda → GND.
Kamera: bilo koji USB priključak; pojavljuje se kao `/dev/video0`.

## Pokretanje

```bash
python3 main.py                  # radni režim, detektor iz config.py
python3 main.py --detector yolo  # jednoprolazni detektor
python3 main.py --save-mode all  # režim prikupljanja skupa podataka
python3 app.py                   # veb server na portu 5000
```

Sistem koristi sistemski Python, jer `opencv` i `gpiozero` dolaze uz Raspberry
Pi OS. Ako se koristi virtuelno okruženje, mora biti napravljeno sa
`python3 -m venv --system-site-packages .venv`.

Prvi minut po pokretanju modul HC-SR501 je u inicijalizaciji i daje do tri
lažna impulsa, pa se okidanja u tom vremenu odbacuju (`WARMUP_S`).

## Moduli

| Datoteka | Uloga |
|----------|-------|
| `main.py` | glavna petlja i merenje vremena t0–t3 |
| `sensor.py` | HC-SR501, čekanje na ivicu, odbacivanje inicijalizacije |
| `camera.py` | otvaranje kamere po okidanju, pražnjenje bafera, upis slike |
| `detectors/` | zajednički interfejs, `hog.py` i `yolo.py` |
| `logger.py` | evidencija u `logs/events.csv`, stanje sistema |
| `led.py` | indikatorska dioda |
| `app.py` | veb sloj (Flask) |
| `eval/` | vrednovanje van linije: `bench.py`, `metrics.py` |

## Vremena koja se beleže

```
t0  uzlazna ivica na GPIO liniji
t1  kadar u memoriji           →  t1 − t0  akvizicija
t2  odluka detektora           →  t2 − t1  zaključivanje
t3  slika i zapis na disku     →  t3 − t0  odziv sistema
```

Svi se upisuju u `logs/events.csv`, jedan red po okidanju.

## Vrednovanje

```bash
# 1. prikupljanje: čuva se svaki kadar posle okidanja, bez iscrtanih okvira
python3 main.py --save-mode all

# 2. slike prebaciti u eval/dataset/images/ i označiti alatom LabelImg
#    (YOLO oblik zapisa, jedna klasa), oznake u eval/dataset/labels/

# 3. oba detektora nad istim skupom
python3 eval/bench.py

# 4. mere i krive
python3 eval/metrics.py --iou 0.5 --target-recall 0.90
```

Rezultati se upisuju u `results/`: nalazi i vremena po detektoru,
`metrics_summary.csv` i kriva preciznosti i odziva `pr_curve.svg`.

Poređenje se vrši **pri uporedivom odzivu**, a ne pri podrazumevanim pragovima,
jer skorovi dva postupka nisu ista veličina: kod YOLO-a je to verovatnoća klase,
a kod HOG-a vrednost odlučujuće funkcije SVM-a.

## Poznata ograničenja

- HOG koristi prozor 64 × 128 piksela, pa osoba niža od 128 piksela u slici ne
  može biti otkrivena. Na 640 × 480 to je 27 % visine kadra.
- Galerija na veb strani učitava sve slike; nema umanjenih prikaza ni stranica.
- Nema brisanja starih snimaka, pa pri dužem radu treba pratiti prostor.
