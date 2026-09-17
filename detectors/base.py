"""Zajednicki oblik nalaza i zajednicki interfejs detektora.

Svaki detektor vraca isto: listu nalaza i trajanje samog zakljucivanja u
milisekundama. Glavna petlja ne zna koji je detektor u pogonu, pa se dva
postupka mogu porediti pod istim uslovima.

Vreme se meri u OSNOVNOJ klasi, a ne u izvedenim, da bi granica merenja bila
dokazano ista za oba postupka: u nju ulazi sve od ulazne slike do konacne
liste nalaza, dakle i priprema ulaza i obrada izlaza, a ne samo prolaz kroz
mrezu odnosno kroz klizni prozor. Kada svaki detektor sam meri svoj deo,
lako se dogodi da jedan u vreme ukljuci obradu izlaza a drugi ne, cime
poredjenje postaje bezvredno."""

import time
from collections import namedtuple

# score je mera poverenja. Za YOLO je to verovatnoca klase (0..1), a za
# HOG vrednost odlucujuce funkcije SVM-a, koja nije verovatnoca i obicno
# je u opsegu 0..2. Vrednosti se zato NE porede izmedju detektora --
# sluze samo za redjanje nalaza i za izbor radne tacke unutar jednog
# postupka (videti 3.6 i 6.6 rada).
Detection = namedtuple("Detection", ["x", "y", "w", "h", "score"])


class Detector:
    """Osnova za detektore. Izvedene klase postavljaju `name` i ostvaruju
    `_detect`, koji vraca listu nalaza; merenje vremena i redjanje po skoru
    obavlja ova klasa, isto za sve detektore."""

    name = "base"

    def _detect(self, frame):
        """Vraca listu Detection za dati kadar."""
        raise NotImplementedError

    def detect(self, frame):
        """Vraca (lista Detection, trajanje_ms)."""
        start = time.perf_counter()
        detections = self._detect(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections, elapsed_ms

    def describe(self):
        """Kratak opis podesavanja, ide u zaglavlje rezultata."""
        return self.name


def filter_by_score(detections, threshold):
    """Radna tacka se bira pragom nad skorom, ne izmenom detektora."""
    return [d for d in detections if d.score >= threshold]
