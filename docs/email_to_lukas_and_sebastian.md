# Entwurf – E-Mail an Lukas & Sebastian

**Betreff:** Volumetric Video Projekt – läuft, lasst uns mal zusammen reinschauen

---

Hi Lukas, hi Sebastian,

ich melde mich, weil ich an einem Projekt sitze, an dem ich mit euch
weiter­denken möchte – und für das ich mit euch zusammen einen kleinen
Förder­antrag stellen würde. Ich erzähl euch erst ganz kurz, *was steht*,
und dann *was ich von euch will*.

## Was steht

Ich habe in den letzten Wochen ein aktuelles Forschungs­paper zu
**volumetrischen Videos** nachvollzogen – das ist die Technik, mit der man
eine reale Szene aus mehreren Kameras aufnimmt und hinterher in VR von
jeder beliebigen Position aus betrachten kann. Bewegtes 3D, im Grunde. Das
Paper (Xu et al., „Temporal Gaussian Hierarchy", Ende 2024) löst das
Problem, dass bisherige Verfahren nach ein paar Sekunden zusammen­brachen –
jetzt sind *minuten­lange* Sequenzen auf einer normalen Gaming-Grafikkarte
machbar.

Die Reproduktion läuft. Auf dem Standard-Benchmark des Original-Datensatzes
(eine 10-Sekunden-Kochszene) erreicht sie nach 50.000 Trainings­schritten
**PSNR 32,08 dB / SSIM 0,970 / LPIPS 0,197** – das **liegt über den
Werten, die das Original-Paper berichtet** (29,44 / 0,945 / 0,214 auf
demselben Benchmark). Heißt: die technische Grundlage steht.

**Ich habe sogar einen kleinen Live-Viewer dazu gebaut**, der das
trainierte Modell über WebSocket an einen Browser streamt. Man bewegt die
Maus, die Szene reagiert in Echtzeit. Das ist der Punkt, an dem ich euch
gerne mit dazu hätte.

## Worum es eigentlich gehen soll

Was mich an dieser Technologie interessiert, ist nicht primär der
Algorithmus, sondern die **künstlerische und gestalterische Frage**, was
man mit so einer „Hologramm-Kamera" eigentlich macht. Volumetrische
Aufnahme bisher hieß: Hollywood-Studio, Sekunden, riesige Kosten. Wenn
das auf Konsumer-Hardware und mehrere Minuten skaliert, **öffnet sich ein
Feld** – aber welches Feld konkret, wissen wir noch nicht. Tanz?
Performance? Dokumentation? Bildung? Etwas ganz anderes? Das ist eine
Frage, die ich nicht am Schreibtisch beantworten kann, sondern nur durch
**ausprobieren mit dem neuen Medium selbst**.

Genau deshalb halte ich den Forschungs­antrag bewusst offen. Ich will
*nicht* jetzt schon festlegen, was wir wissenschaftlich publizieren oder
welche Erweiterung wir bauen. Ich will die Hardware bekommen, mit der wir
das Medium *anfassen* können – und dann zu dritt schauen, in welche
Richtung sich das tatsächlich entwickelt.

## Was ich mit euch zusammen testen möchte

Bevor wir den Antrag formulieren, hätte ich gerne dass wir uns einmal
zu dritt zusammen­setzen und das aktuelle System gemeinsam ausprobieren.
Konkret bräuchten wir dafür:

- **einen Rechner mit einer halbwegs aktuellen Grafikkarte** (RTX 3060
  aufwärts genügt – meiner kann ich mitbringen, oder wir nutzen einen bei
  euch),
- **eine Meta Quest** (idealerweise Quest 3, geht auch Quest 2). Lukas,
  ihr habt eine in der Werkstatt, oder?

Damit können wir das trainierte Modell live ansehen, mit den Parametern
spielen, schauen wie es sich anfühlt – und dann gemeinsam ein Gefühl
entwickeln, *wofür* wir das eigentlich aufnehmen wollen.

## Der Antrag

Der Antrag selbst wäre klein und Hardware-fokussiert (≈ 10.500 €):

- **zehn 360°-Kameras (Insta360 X5)** plus Stative, Speicher, Sync-Klappe:
  ein Aufnahme-Rig, mit dem wir unsere erste eigene 3-minütige
  volumetrische Sequenz aufnehmen können (≈ 8.000 €),
- **eine NVIDIA RTX 5090** für die Rekonstruktion (≈ 2.500 €). Mein
  aktueller Rechner hat eine 3090 mit 24 GB VRAM – das reicht knapp für
  40-Sekunden-Sequenzen. Für die geplanten 3-Minuten-Aufnahmen brauchen
  wir die 32 GB der 5090, sonst läuft uns der Speicher voll bevor der
  Trainings­lauf durch ist.

Was die Sequenz inhaltlich werden soll, lassen wir bewusst offen – das
ergibt sich, wenn wir gemeinsam mit dem Medium experimentieren.

Keine Personal­kosten, keine VR-Endgeräte im Antrag (die geliehene Quest
reicht für den Anfang).

## Wenn ihr Zeit habt

Würde mich freuen, wenn wir uns nächste oder über­nächste Woche treffen
können – 90 Minuten reichen für eine erste Test­sitzung. Sagt mir, wann
es bei euch passt und wo wir am besten Rechner + Quest zusammenkriegen.

Falls ihr vorher reinlesen wollt, liegen alle Dokumente unter `docs/`
im Projekt­ordner (Konzeptpapier, technische Doku, Literatur, ein paar
fertig gerenderte Bilder). Aber das ist optional – wir können das
genauso gut anhand des Live-Viewers besprechen.

Schönen Gruß,
[Name]
