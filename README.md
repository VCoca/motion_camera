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
| `notifier.py` | obaveštenje poštom, vremenska zabrana, podešavanja |
| `supervisor.py` | pokretanje i zaustavljanje glavne petlje sa strane |
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

## Upravljanje sa veb strane

Sve radnje mogu se izvesti iz pregledača, bez terminala:

| Radnja | Gde |
|--------|-----|
| pokretanje i zaustavljanje nadzora | Upravljanje sistemom |
| izbor detektora (`hog` / `yolo`) | isto, pri pokretanju ili preko „Primeni i ponovo pokreni" |
| režim čuvanja (`person` / `all`) | isto |
| prag poverenja | isto, prazno polje znači bez praga |
| dnevnik rada glavne petlje | isto, „Dnevnik rada" |
| adresa primaoca, prekidač i razmak poruka | Obaveštenje elektronskom poštom |
| probna poruka | isto, dugme „Pošalji probnu poruku" |
| preuzimanje evidencije | dugmad na dnu |
| brisanje evidencije i slika | isto |

Veb proces ne poziva glavnu petlju nego je pokreće kao **poseban proces**,
jer petlja drži GPIO linije i kameru dok server usužuje zahteve. Broj procesa
se pamti u `logs/main.pid`, pa stanje preživi i ponovno pokretanje servera.

Detektor se bira pri pokretanju, pa njegova promena znači zaustavljanje i
ponovno podizanje petlje — dugme to radi u jednom koraku.

**GPIO linija se ne zaključava**, pa se dve petlje mogu pokrenuti uporedo i
otimati se o kameru. Zato strana traži glavnu petlju među procesima, a ne samo
u svom zapisu: petlja pokrenuta iz terminala prikazuje se kao „pokrenuta izvan
ove strane" i pokretanje druge se odbija.

Strana nema prijavu, pa svako na istoj mreži može da pokrene i zaustavi
sistem. Za rad u zatvorenoj kućnoj mreži to je prihvatljivo; za bilo šta drugo
bi trebalo dodati proveru identiteta.

## Obaveštenje elektronskom poštom

Kada detektor nađe osobu, sistem može da pošalje poruku sa slikom u prilogu.
Adresa primaoca, prekidač i najkraći razmak između poruka zadaju se na veb
strani i čuvaju u `settings.json`.

Pristupni podaci naloga sa koga se šalje drže se **odvojeno**, u `smtp.json`,
i nikada se ne prikazuju na strani:

```bash
cp smtp.example.json smtp.json
nano smtp.json          # uneti nalog i lozinku za aplikaciju
```

Za Gmail je potrebna lozinka za aplikaciju (dvostepena provera mora biti
uključena), ne obična lozinka naloga. Umesto datoteke mogu se zadati i
promenljive okruženja `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`
i `SMTP_FROM`.

Tri stvari koje određuju izvedbu:

- **Slanje ide u zasebnoj niti.** SMTP razmena traje i po nekoliko sekundi, a
  vreme odziva sistema (t₃ − t₀) meri se za rad, pa slanje ne sme da uđe u taj
  put. Izmereni odziv ostaje isti bez obzira na to da li se poruka šalje.
- **Vremenska zabrana**, podrazumevano 5 minuta, sprečava da neprekidan pokret
  pošalje desetine poruka. Detekcije u tom vremenu se broje i njihov broj ulazi
  u sledeću poruku.
- **Neuspeh slanja ne obara nadzor.** Greška se zapisuje i prikazuje na strani,
  a petlja nastavlja da radi.

## Poznata ograničenja

- HOG koristi prozor 64 × 128 piksela, pa osoba niža od 128 piksela u slici ne
  može biti otkrivena. Na 640 × 480 to je 27 % visine kadra.
- Galerija na veb strani učitava sve slike; nema umanjenih prikaza ni stranica.
- Nema brisanja starih snimaka, pa pri dužem radu treba pratiti prostor.
