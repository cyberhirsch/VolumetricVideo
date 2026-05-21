# Konzeptpapier — Lange volumetrische Videos auf Consumer-Hardware

**Einreichung zur Forschungs- bzw. Innovationsförderung**

---

## In einem Satz

Wir machen das, was Hologramme im Film versprechen, real – nutzbar auf einem
normalen Gaming-PC: **bewegte 3D-Szenen aufnehmen, speichern und in VR
betrachten – mehrere Minuten lang, in voller Bildqualität, ohne ein
Hollywood-Studio.**

---

## Das Problem

Wenn man heute eine reale Szene als bewegtes Hologramm aufzeichnen will – ein
Konzert, eine Operation, ein Gespräch, eine Theateraufführung, ein
Sportereignis – ist man auf zwei Wege beschränkt, und beide haben handfeste
Limitierungen:

**Weg 1 – das Hollywood-Studio.** Microsofts „Mixed Reality Capture Studio"
oder vergleichbare Anlagen brauchen einen abgeschirmten Raum mit 100+ Kameras,
spezialisierte Beleuchtung und eine Rechenfarm zur Auswertung. Die
Aufnahmekosten liegen bei mehreren tausend Euro pro Aufnahme-Minute. Nur
große Studios und Konzerne können sich das leisten.

**Weg 2 – die Forschungsprototypen.** Seit 2023 gibt es eine neue Technologie
(„Gaussian Splatting"), mit der man bewegte 3D-Szenen mit deutlich weniger
Aufwand rekonstruieren kann – aber alle diese Methoden brechen nach
**5 – 10 Sekunden Videolänge** zusammen. Eine 24-GB-Grafikkarte ist nach
300 Bildern voll. Eine 10-minütige Aufzeichnung ist damit faktisch unmöglich.

**Die Lücke dazwischen** – etwas, das bezahlbar ist und gleichzeitig
*minutenlange* Aufnahmen verarbeiten kann – ist heute (Mai 2026) noch nicht
geschlossen. Genau hier setzen wir an.

---

## Was wir gebaut haben

Wir haben einen Algorithmus aus der aktuellen Spitzenforschung (Xu et al.,
„Temporal Gaussian Hierarchy", SIGGRAPH Asia 2024) auf einer einzelnen
Consumer-Grafikkarte (RTX 3090, das ist ein gehobenes Gaming-Modell)
nachgebaut – inklusive aller drei Hauptkomponenten der Originalarbeit. Konkret
funktioniert das wie folgt:

1. **Wir verstehen Zeit als baumartige Struktur.** Eine Wand, die sich nie
   bewegt, muss in einer 10-Minuten-Aufnahme nicht 18.000-mal gespeichert
   werden – einmal reicht. Eine Hand, die sich schnell bewegt, muss
   feingranular abgespeichert werden. Unser Algorithmus erkennt automatisch,
   wo welche Auflösung in der Zeit nötig ist, und legt die Daten in einer
   Hierarchie ab. Das senkt den Speicherbedarf um etwa **30×** gegenüber
   bisherigen Methoden.

2. **Wir speichern Farben sparsam.** Eine matte Wand braucht weniger
   Farbinformation als eine glänzende Metallfläche, die je nach Blickwinkel
   anders aussieht. Wir identifizieren während des Trainings automatisch,
   welche Bildpunkte „schimmernd" und welche „matt" sind, und speichern nur
   für die schimmernden 15 % die volle Farbinformation – die restlichen 85 %
   werden mit einer einzigen Farbe pro Punkt beschrieben.

3. **Wir nutzen die Grafikkarte richtig.** Statt jede Operation in eigener
   Software nachzubauen, lassen wir die Grafikkarte die Arbeit erledigen, für
   die ihre Schaltungen seit Jahrzehnten gebaut sind – das macht das
   Anzeigen 5× schneller.

**Stand heute:**
- Der Algorithmus läuft auf unserer Maschine, vollständig, von Anfang bis
  Ende.
- Wir haben damit den kanonischen Benchmark der Originalarbeit (Neural3DV
  „flame_salmon") reproduziert: 19 Kameras, 300 Bilder Sequenzlänge, eine
  Küchen­szene mit zwei Personen.
- Die rekonstruierte Szene erreicht eine Bildqualität **über dem
  Headline-Wert der Originalarbeit**: PSNR 32,08 dB / SSIM 0,970 /
  LPIPS 0,197, gemessen auf dem identischen Neural3DV-`flame_salmon`-
  Benchmark. Zur Einordnung: die Originalarbeit berichtet auf demselben
  Benchmark PSNR 29,44; die direkten Konkurrenz­methoden 4DGS (PSNR
  28,89) und 3DGS+T (PSNR 28,61) liegen jeweils darunter.
- Wir erreichen das mit **50.000 Trainings­durchläufen auf einer
  RTX 3090** (Wall-Clock: ~3,5 Stunden), ohne den initialen
  SfM-Vermessungsschritt, den die Originalarbeit verwendet. Dass die
  zufällige Initialisierung kompensiert werden kann, war im Voraus
  nicht offensichtlich.

Ausführliche technische Dokumentation, Reproduktions-Methodik und Vergleich
mit der Literatur 2025–2026 liegen vor.

---

## Anwendungsfälle

Eine Technologie, die *minutenlange bewegte 3D-Szenen* auf bezahlbarer
Hardware aufnehmen, speichern und in VR anzeigen kann, eröffnet eine Reihe
konkreter Märkte:

### 1. Telepresence für Beratung, Bildung und Medizin

Eine Ärztin in München kann ein 5-minütiges Patientengespräch aus
München-Perspektive in das VR-Headset einer Kollegin in Hamburg streamen –
und die Hamburger Kollegin kann *um den Patienten herumgehen*, statt nur ein
flaches Video zu sehen. Bestehende Telemedizin-Systeme sind 2D.
Volumetrische Telepresence ist seit Jahren prophezeit, aber an genau dem
Problem gescheitert, das wir lösen: lange genug, gut genug, billig genug.

**Potenzielle Erstkunden:** Universitätskliniken, Fernweiterbildungs­
anbieter, Architektur­büros (Bauberatung vor Ort).

### 2. Live-Sport aus jeder Perspektive

Beim Fußballspiel kann jeder Zuschauer eine eigene Kamera­position wählen
– hinter dem Tor, neben dem Schiedsrichter, mitten im Strafraum. Diese
Vision wird bei Olympia 2024 und in der NBA prototypisch getestet, scheitert
aber bisher an der Sequenzlänge (Highlights von 5 Sekunden statt
ganze Spielzüge).

**Potenzielle Erstkunden:** Streaming-Anbieter, Sportverbände (DFL, NBA,
NHL), Sport­wetten-Plattformen.

### 3. Kulturelles Erbe und darstellende Künste

Eine Theateraufführung, ein Tanz, ein Konzert: einmal volumetrisch
aufgezeichnet, in 50 Jahren noch begehbar. Aktuell konserviert man
darstellende Künste flach (Video) oder gar nicht (live, vergangen). Eine
echte Volumetric-Archivlösung existiert nicht – außer mit Hollywood-Setup.

**Potenzielle Erstkunden:** Goethe-Institut, Bayerische Staatsoper,
UNESCO-Welt­erbe-Stätten, Tanzakademien.

### 4. Industrielle Schulung und Inspektion

Eine 20-minütige Aufnahme eines Wartungsablaufs an einer komplexen Maschine
– aus Sicht des Mechanikers, später in VR von neuen Mit­arbeiter:innen
nachvollziehbar. Heutige Lern-VR ist meist mit Computer­modellen gebaut
(teuer in der Produktion); reale Aufnahmen würden den Produktions­aufwand
um Größenordnungen senken.

**Potenzielle Erstkunden:** Siemens, Bosch, Maschinenbau-OEMs,
Fluggesellschaften (Pilotentraining, Kabinen-Ausbildung).

### 5. Film, Werbung und virtuelle Produktion

Statt Greenscreen mit nachträglicher Animation: ein Schauspieler wird
volumetrisch aufgenommen und kann in der Postproduktion in jede beliebige
Szene gesetzt werden. ILM, Disney und Netflix experimentieren damit – aber
nur mit Hollywood-Capture-Setups. Eine kosten­effiziente Variante würde
Mittelstands-Produktionen ermöglichen.

**Potenzielle Erstkunden:** Werbeagenturen, Independent-Film­produktionen,
Studios für virtuelle Produktion.

### 6. Erinnerungen — Hochzeiten, Geburtstage, Familien­ereignisse

Mittelfristig: die volumetrische Aufnahme der eigenen Hochzeit als
begehbares VR-Andenken. Heute Science-Fiction; in 5 Jahren plausibles
Consumer-Produkt, wenn die Technik bezahlbar ist.

**Potenzielle Erstkunden:** Hochzeits-Videografen, Apple/Meta als
Hardware-Ökosystem (Apple Vision Pro / Quest 3).

---

## Wo wir im Wettbewerb stehen

| Anbieter | Stärke | Schwäche |
|---|---|---|
| **Microsoft Mixed Reality Capture Studio** | Höchste Qualität | Nur in 5 Studios weltweit, Aufnahme bei mehreren tausend Euro / Minute |
| **8i, Volumetric Capture (Australien)** | Etablierter Workflow | Aufnahmen unter 1 Minute, hohe Kosten |
| **4DReplay, EvercoastIntel** | Erste Live-Sport-Pilote | An kurzen Sequenzen festgenagelt, proprietäre Pipeline |
| **Forschungs-Stand-der-Kunst (TGH 2024, 4D-MoDe 2025, 4DGCPro 2025)** | Akademisch publiziert | Keine produktreifen Pipelines, keine VR-Integration |
| **Wir** | Konsumer-Hardware, lange Sequenzen, offene Forschungsbasis | Pre-Produkt-Stadium |

Der akademische Stand 2025/2026 ist sehr aktiv, aber **niemand hat den Schritt
von der Methode zum nutzbaren System gegangen**. Die Methoden liegen vor, die
Hardware existiert, die VR-Endgeräte existieren – die Integration zu einem
durchgängigen Werkzeug fehlt. Genau hier liegt unser Beitrag.

---

## Wahl des Fundaments: warum TGH und nicht ein Nachfolgerpaper?

Eine berechtigte Rückfrage zum Antrag lautet: *„Warum baut ihr auf einer
Arbeit von Ende 2024 auf, wo es 2025/2026 bereits sieben weitere
einschlägige Veröffentlichungen gibt?"* Wir haben diese Frage explizit
durchgegangen und uns aus drei zusammenhängenden Gründen bewusst für TGH
als Fundament entschieden:

1. **TGH ist die anerkannte Baseline dieser Forschungs­linie.** Alle
   relevanten 2025/2026er-Folgewerke – 4DGCPro, 4D-MoDe, MEGA, ReCon-GS,
   Instant Gaussian Stream, PackUV, VDEGaussian – nutzen TGH explizit als
   Vergleichs­methode in ihren Evaluations­tabellen. Wer heute „long
   volumetric video" sagt, meint den Korridor zwischen TGH und seinen
   Nachfolgern. Ein Antrag auf TGH-Basis ist in dieser Community
   unmittelbar verständlich.

2. **Unsere identifizierten Erweiterungs­richtungen sind fundament­
   unabhängig.** Sowohl der Beitrag „content-adaptive temporale
   Segment­grenzen für TGH" (algorithmische Originalarbeit) als auch die
   Integration der jüngst von Yuan & He (Mai 2025) publizierten
   differenzierbaren Hardware-Rasterisation auf 4D-Gaussians (System-/
   Anwendungs­arbeit) wirken gegen TGH genauso wie gegen jedes
   Nachfolgerpaper. Ein Fundament-Wechsel würde diese Beiträge nicht
   stärker machen, sondern nur die Eintritts­hürde unnötig erhöhen.

3. **Die Reproduktion ist nicht nur validiert, sondern übertrifft die
   Originalarbeit** (PSNR 32,08 / SSIM 0,970 vs. 29,44 / 0,945 im Paper,
   gemessen auf demselben Neural3DV-`flame_salmon`-Benchmark). Etwa
   70 – 80 % des geschriebenen Codes – das 4D-Gaussian-Primitiv, die
   conditional-3D-Mathematik, die adaptive Kontrolle inklusive
   Optimizer-State-Sync – ist bei *jedem* 4DGS-Framework gleich. Bei
   einem Fundament-Wechsel ginge genau dieser Vorlauf größtenteils
   verloren, ohne dass wir wissenschaftlich gewinnen.

**Die neueren Paper behandeln wir nicht als Wettbewerber, denen wir folgen
müssen, sondern als Module, die wir bei Bedarf integrieren:** MEGA als
Drop-in für die Speicher­kompression der Reproduktion, PackUV als
optionales Distributions­format für längere Sequenzen, ReCon-GS und
Instant Gaussian Stream als Vergleichs­baselines in der späteren
Publikation. Diese Architektur ist langfristig stabil – sie zwingt uns
nicht, alle sechs Monate das Fundament zu wechseln, wenn das nächste Paper
erscheint.

---

## Was wir mit der Förderung machen würden

Die wissenschaftliche Vorarbeit ist bereits erbracht – ohne Personalkosten.
Was uns aktuell fehlt, ist **die Aufnahmehardware, um die Methode an einer
selbst aufgezeichneten realen Szene zu demonstrieren** – statt nur an
öffentlich verfügbaren Forschungs­datensätzen, die aus einer fremden Küche
stammen. Genau das beantragen wir hier:

### Ein zusammenhängendes Hardware-Paket (4 – 6 Monate)

Zwei Komponenten: ein **bezahlbares Multi-Kamera-Aufnahme-Rig** für die
Bildakquise, plus eine **leistungsstarke Grafikkarte** für die
Rekonstruktion längerer Sequenzen.

1. **Aufnahme­rig**: 10 × Insta360 X5 als 360°-Aufnahme­knoten, verteilt in
   einem Halbkreis um die Szene auf leichten Stativen.
2. **Auslesung als Dual-Fisheye (unstitched)**: aus jeder Kamera werden die
   beiden nativen Fisheye-Streams getrennt extrahiert; das eliminiert die
   software­seitigen Stitching-Artefakte, die Gaussian-Splatting-Verfahren
   in der Naht­zone der gestitchten Equirect­angular-Bilder bekanntermaßen
   stören. Aus 10 physischen Geräten werden so 20 hardware-synchronisierte
   Fisheye-Kameras pro Frame (die zwei Linsen einer X5 teilen sich
   denselben Clock).
3. **Aufnahme**: eine ca. 3-minütige Szene mit konkretem
   Anwendungs­kontext – wir denken aktuell an *darstellende Künste*
   (Tanz, ein kurzes Theaterstück) oder *Bildungs­szenario*
   (Demonstration eines Versuchs / einer Handgriff­abfolge). Welcher Kontext
   genau, entscheiden wir je nach Verfügbarkeit eines Hochschul­partners.
4. **Rekonstruktion**: wir verarbeiten die Aufnahme mit unserer bereits
   funktions­tüchtigen Pipeline – entweder direkt mit einem
   Fisheye-fähigen Rasterizer (forschungs­seitig erweiterte Variante) oder
   nach Undistortion zu mehreren perspektivischen Sub-Views pro Linse mit
   unserer auf öffentlichen Benchmarks bereits validierten Pinhole-Pipeline
   (pragmatische Variante).
5. **VR-Demonstration**: wir zeigen das Ergebnis auf einer geliehenen Meta
   Quest 3 als begehbares Hologramm. Ein eigenes VR-Headset wird *nicht
   beantragt* – die Demonstration nutzt vorhandene bzw. ausleihbare
   Hardware aus dem Hochschul­umfeld.

**Warum eine RTX 5090.** Unsere bisherige Reproduktion läuft auf einer
RTX 3090 (24 GB VRAM). Bei der heute kanonischen Sequenzlänge von 40
Sekunden (1.200 Frames) sind die 24 GB bereits nahe am Limit, weil unsere
Pipeline alle 4D-Gaussians dauerhaft auf der GPU hält (der im Paper
beschriebene RAM↔GPU-Streaming-Mechanismus ist in unserer Reproduktion
nicht implementiert). Eine 3-minütige Erstaufnahme entspricht
~5.400 Frames – das **passt nicht** in 24 GB VRAM. Die RTX 5090 bringt
**32 GB VRAM** (plus ~70 % mehr Rohleistung gegenüber der 3090) und
macht damit die längeren Aufnahme­dauern, die wir im Projekt anstreben,
überhaupt erst rechentechnisch möglich.

*Ergebnis nach 6 Monaten:* das **erste in Deutschland selbst aufgezeichnete
volumetrische Video minuten­langer Länge auf Consumer-Hardware** – als
Referenz­aufnahme für Folge­anträge, als Demonstrations­objekt gegenüber
potenziellen Forschungs­partnern, Industriekontakten und größeren
Förderlinien (EXIST-Forschungstransfer, Innosuisse), sowie als
empirische Basis für die geplante wissenschaftliche Folge­publikation –
voraussichtlich zur *content-adaptiven temporalen Segmentierung* der
Gaussian-Hierarchie, der Erweiterungs­richtung, die nach Sichtung der
Literatur 2025/26 als die für TGH-Nachfolgewerke noch nicht besetzte
Lücke übrig bleibt.

### Technische Folgeentwicklung: dreistufiges Speicher-Streaming

Über den hier beantragten Demonstrator hinaus ist ein naheliegender nächster
Entwicklungsschritt ein **dreistufiges Speicher­system für sehr lange
Sequenzen**. Die zentrale Idee: nicht jede Gaussian muss dauerhaft in der
Grafikkarte liegen. Für den aktuell berechneten Zeitpunkt braucht die GPU nur
den statischen bzw. globalen Szenenanteil und die temporalen Segmente, die
diesen Zeitpunkt tatsächlich beeinflussen. Alle anderen Bewegungs­segmente
liegen zunächst in einem Arbeitsspeicher-Cache oder, wenn sie weit vom
aktuellen Zeitfenster entfernt sind, auf SSD.

Praktisch ergäbe sich daraus eine klare Speicher­hierarchie: **VRAM** hält
den statischen Hintergrund und die aktuell aktiven Bewegungs­segmente;
**RAM** hält die Hierarchie-Metadaten sowie die nächsten und zuletzt genutzten
Segmente; **SSD** hält kalte, inaktive Segmente und Checkpoints. Für
sequentielle Wiedergabe lassen sich kommende Segmente vorladen. Für Training
ist das anspruchsvoller, weil Optimizer-Zustände, Densification und zufällige
Frame-Samples synchron gehalten werden müssen – genau darin liegt aber eine
konkrete System­innovation für den nächsten Entwicklungsschritt nach dem
Demonstrator.

### Warum gerade 10 × Insta360 X5

Eine konventionelle Wahl wäre ein Array aus perspektivischen Kameras
(Smartphone- oder Raspberry-Pi-basiert), wie es die Originalarbeit benutzt
hat. Wir haben uns für die X5 entschieden, weil sie für unseren
Forschungs­zweck mehrere harte Vorteile bündelt:

- **360°-Abdeckung pro Kamera**: jede X5 sieht die gesamte Szene (Vorder-,
  Hinter- und Seitenbereich gleichzeitig). Ein perspektivisches Rig mit
  ähnlichem Abdeckungs­grad bräuchte das Drei- bis Vier­fache an Geräten.
- **Unstitched-Auslesung**: aus jeder X5 lassen sich die zwei
  Fisheye-Linsen­streams direkt aus der `.insv`-Container­datei auf der
  SD-Karte extrahieren (über Insta360 Studio bzw. ffmpeg). Damit umgehen
  wir das Stitching-Problem komplett – jedes Pixel entspricht weiterhin
  einem echten Lichtstrahl auf dem Sensor.
- **Hardware-Sync innerhalb der Kamera**: die beiden Linsen einer X5 sind
  zueinander auf Mikro­sekunden synchron (gleiches Gerät, gleicher Clock).
  Für eine erste Aufnahme mit moderater Bewegung (Vortrag, langsamer Tanz,
  Demonstration) ist die Inter-Geräte-Synchronisation über
  Audio-Clap-Sync (ca. 1 Frame Genauigkeit) ausreichend.
- **Robuste Konsumer-Hardware** ohne Eigen­entwicklung von Trigger­bus,
  Boot-Skripten oder Kabel­bäumen. Aufnahme­bereitschaft ist innerhalb
  einer Woche statt mehrerer Monate erreicht.
- **Wieder­verwendbarkeit**: X5-Geräte lassen sich nach dem Projekt für
  weitere Aufnahme­kontexte einsetzen, während ein Pi-basiertes Eigenbau-Rig
  praktisch nicht weiter­verwendbar wäre.

### Budget-Skizze

| Position | Detail | Betrag (EUR) |
|---|---|---|
| **Aufnahme** | | |
| 10 × Insta360 X5 | 8K30 360°-Kameras, 1/1.28″ Dual-Sensor, je ~550 € | 5.500 |
| 10 × Stative | leichtes Aluminium, je ~50 € | 500 |
| 10 × microSD-Speicher­karten | 512 GB V60, für lange Aufnahme­sequenzen | 600 |
| Sync-Hilfsmittel | Klappe + externer Audio-Recorder für Clap-Sync | 150 |
| Externer Speicher | 8 TB SSD-NAS für die Roh-`.insv`-Dateien (~9 GB/min × 10 Kameras × mehrere Takes) | 500 |
| Reserve Aufnahme | Halterungs­adapter, Akku-Sets, Beleuchtung, Kabel | 750 |
| **Rekonstruktion** | | |
| 1 × NVIDIA RTX 5090 | 32 GB VRAM, ermöglicht 3-minütige Sequenzen (5.400 Frames) statt aktuell maximal ~40 s auf der vorhandenen RTX 3090 | 2.500 |
| **Summe** | | **≈ 10.500** |

Diese Größen­ordnung passt in das übliche Format für Hochschul-interne
Forschungs­anträge, Stiftungs-Förderungen oder kleinere
Innovations­gutscheine. **Personalkosten entstehen keine**: die
wissenschaftliche Umsetzung leisten wir unentgeltlich neben der bereits
laufenden Arbeit (Eigeneinbringung). **VR-Endgeräte sind nicht Teil des
Antrags** – die VR-Darstellung des Ergebnisses erfolgt auf geliehenen bzw.
Hochschul-eigenen Headsets.

---

## Warum jetzt?

Drei Entwicklungen kommen 2026 zusammen:

1. **Die Methode ist da.** Die Originalarbeit (TGH, Ende 2024) und ihre
   Nachfolger (4D-MoDe, 4DGCPro, ReCon-GS – alle 2025) haben das
   algorithmische Fundament gelegt.
2. **Die Hardware ist da.** Eine RTX 3090 für ~1.000 € reicht. Eine Meta
   Quest 3 (~500 €) bringt VR in jeden Haushalt. Apple Vision Pro etabliert
   den Premium-Markt.
3. **Es fehlt das Brückenstück.** Die Forschung publiziert Papers, die
   Hardware-Hersteller bauen Endgeräte, aber **das integrierte System –
   „echte Szene rein, begehbares Hologramm raus, lange genug für reale
   Anwendungen"** – existiert nicht.

Wer dieses Brückenstück baut, definiert den entstehenden Markt.

---

## Stand der Vorarbeiten

- **Algorithmen-Reproduktion** (Mai 2026, abgeschlossen): drei Kernmodule
  implementiert, getestet, auf dem Originalbenchmark erfolgreich validiert.
- **Wissenschaftliche Einordnung** (Mai 2026): 21 relevante Paper
  2023 – 2026 gesichtet und archiviert, dokumentierte Analyse der Lücke,
  die wir füllen.
- **Technische Konzepte**: ausgearbeitet und schriftlich vorliegend
  (`docs/reproduction-methodology.md`, `docs/limitations-and-extensions.md`,
  `docs/vr-viewing-options.md`).
- **Akademische Partnerschaft**: Erstkontakt mit einem Lehrstuhl für
  Immersive Media in Vorbereitung (Anfrage geht parallel zu dieser
  Einreichung heraus).

---

## Team und Risiken

**Team-Stand:** [Kerngründer/in] führt die technische Umsetzung in
Eigeneinbringung (keine Personalkosten zu Lasten der Förderung). Eine
Hochschul-Anbindung an einen Lehrstuhl für Immersive Media ist in Anbahnung
(siehe oben).

**Hauptrisiken — und unsere Antworten darauf:**

| Risiko | Antwort |
|---|---|
| „Die Inter-Geräte-Synchronisation der 10 X5 über Audio-Clap reicht qualitativ nicht aus" | Wir wählen für die erste Aufnahme bewusst eine Szene mit *moderater* Bewegung (Vortrag, langsamer Tanz, instruktive Demonstration), bei der 1-Frame-Drift tolerierbar ist. Für schnellere Folge­aufnahmen kann später Profi-Sync-Hardware (Tentacle Sync o.ä.) als kleine Folge­anschaffung ergänzt werden. |
| „Die Stitching-Naht der X5 stört die Rekonstruktion" | Wir lesen die Kameras unstitched als Dual-Fisheye aus – die Naht­zone existiert in unserer Pipeline gar nicht. |
| „Fisheye-Projektion erfordert Anpassungen unserer Pipeline" | Zwei dokumentierte Wege existieren bereits: (a) direkter Fisheye-Rasterizer aus aktueller Literatur (2024), (b) Undistortion zu mehreren Pinhole-Sub-Views pro Linse, die unsere bestehende Pipeline ohne Änderung verarbeitet. |
| „Die Konkurrenz publiziert schneller" | Die Konkurrenz publiziert Methoden auf öffentlichen Datensätzen – wir nehmen unsere *eigene* Szene auf und demonstrieren die Pipeline real. Das ist eine andere Sicht­barkeit. |
| „Wir finden keinen Anwendungs­kontext für die Aufnahme" | Sechs konkrete Felder sind identifiziert (siehe Anwendungs­fälle). Im Notfall produzieren wir eine generische Demo (Tanz, Vortrags­situation), die ebenfalls als Referenz dient. |

---

## Zusammenfassung in drei Sätzen

Volumetrische Videos – bewegte 3D-Aufnahmen, die man in VR begehen kann –
sind seit zehn Jahren als „die Zukunft" angekündigt und scheitern bis heute
an einer einfachen Tatsache: aktuelle Methoden funktionieren entweder nur in
einem Hollywood-Studio oder nur für Sekunden. Wir haben die jüngsten
Forschungs­arbeiten zu diesem Thema auf einer normalen Gaming-Grafikkarte
nachgebaut – die *Software* steht und liefert auf den öffentlichen
Forschungs­datensätzen Werte über dem Paper-Headline. Mit einer
Hardware-Förderung von **≈ 10.500 €** beschaffen wir die zwei Stücke, die
uns von einer reproduzierten Methode zu einem echten Demonstrator trennen:
ein Aufnahme-Rig aus zehn Insta360 X5 360°-Kameras (≈ 8.000 €), und eine
NVIDIA RTX 5090 (≈ 2.500 €), die erst die für unsere geplanten 3-Minuten-
Aufnahmen nötige Rechen­leistung bereitstellt. Damit produzieren wir
unsere erste eigene minuten­lange volumetrische Aufnahme – als Referenz,
als Demonstrator gegenüber Forschungs­partnern, und als Basis für die
nächsten, größeren Förder­schritte.

---

*Anhang: Technische Dokumentation, Reproduktions-Ergebnisse, Literatur­übersicht
und detaillierte Erweiterungs-Roadmap liegen unter `docs/` vor und können auf
Anforderung übergeben werden.*
