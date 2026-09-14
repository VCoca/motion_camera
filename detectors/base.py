"""Zajednicki oblik nalaza i zajednicki interfejs detektora.

Svaki detektor vraca isto: listu nalaza i trajanje samog zakljucivanja u
milisekundama. Glavna petlja ne zna koji je detektor u pogonu, pa se dva
postupka mogu porediti pod istim uslovima."""

from collections import namedtuple

# score je mera poverenja. Za YOLO je to verovatnoca klase (0..1), a za
# HOG vrednost odlucujuce funkcije SVM-a, koja nije verovatnoca i obicno
# je u opsegu 0..2. Vrednosti se zato NE porede izmedju detektora --
# sluze samo za redjanje nalaza i za izbor radne tacke unutar jednog
# postupka (videti 7.6 istrazivackog rada).
Detection = namedtuple("Detection", ["x", "y", "w", "h", "score"])


class Detector:
    """Osnova za detektore. Izvedene klase postavljaju `name` i
    ostvaruju `_detect`."""

    name = "base"

    def _detect(self, frame):
        raise NotImplementedError

    def detect(self, frame):
        """Vraca (lista Detection, trajanje_ms)."""
        raise NotImplementedError

    def describe(self):
        """Kratak opis podesavanja, ide u zaglavlje rezultata."""
        return self.name


def filter_by_score(detections, threshold):
    """Radna tacka se bira pragom nad skorom, ne izmenom detektora."""
    return [d for d in detections if d.score >= threshold]
