# SwissGPC
This repository holds the official implementation of the SwissGPC (Swiss German Podcast Corpus) pipeline used to weakly label data collected from
YouTube and the Swiss Broadcasting Corporation (SRG / SRF). As we do not possess any rights to the collected data, it is not
possible to publish the annotated dataset itself. Instead, we publish the data pipeline that downloads, transcribes and
prepares the data for usage in, for example, fine-tuning a model for Voice Adaptation TTS.

## Podcasts
The utilized podcasts with their links are provided here with information about raw and cleaned audio size in hours. As outlined above: We do not possess rights or have ownership of these podcasts and as such, 
any changes on the platforms they are hosted on are out of our control. Meaning constant updates of changing hyperlinks, partial or complete removal, and similar changes 
do not fall within the scope of this repository. We will try to provide a general overview of the availability but cannot guarantee to do so in real-time. The podcasts 
were downloaded over a period of time spanning from September 2024 to March 2025 and as such my not reflect the actual audio lengths of the podcasts on time of download.

| **SRF Podcast Name**                                                                                                                                   | **Raw (h)** | **Clean (h)** | **vSwissGPC** |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|---------------|---------------|
| [#SRFglobal](https://www.srf.ch/sendungen/srfglobal)                                                                                                   | 36.97       | 33.63         | v1.0          |
| [100 Sekunden Wissen](https://www.srf.ch/audio/100-sekunden-wissen)                                                                                    | 186.75      | 152.12        | v1.0          |
| [BuchZeichen](https://www.srf.ch/audio/buchzeichen)                                                                                                    | 365.10      | 305.62        | v2.0          |
| [Debriefing 404](https://www.srf.ch/audio/debriefing-404)                                                                                              | 243.15      | 195.29        | v1.0          |
| [Digital Podcast](https://www.srf.ch/audio/digital-podcast)                                                                                            | 434.56      | 396.59        | v1.0          |
| [Dini Mundart](https://www.srf.ch/audio/dini-mundart)                                                                                                  | 39.28       | 34.84         | v1.0          |
| [Einfach Politik](https://www.srf.ch/audio/einfach-politik)                                                                                            | 40.69       | 38.07         | v2.0          |
| [Espresso](https://www.srf.ch/audio/espresso)                                                                                                          | 565.84      | 500.50        | v2.0          |
| [Focus](https://www.srf.ch/audio/focus)                                                                                                                | 807.08      | 630.22        | v2.0          |
| [Gast am Mittag](https://www.srf.ch/audio/gast-am-mittag)                                                                                              | 34.07       | 30.43         | v1.0          |
| [Geek-Sofa](https://www.srf.ch/audio/geek-sofa)                                                                                                        | 314.01      | 267.16        | v1.0          |
| [Input](https://www.srf.ch/audio/input)                                                                                                                | 714.13      | 602.91        | v2.0          |
| [SRF-Wissen](https://www.srf.ch/audio/srf-wissen)                                                                                                      | 44.78       | 39.17         | v1.0          |
| [Krimi](https://www.srf.ch/audio/krimi)                                                                                                                | 240.80      | 176.05        | v2.0          |
| [Kultur-Talk](https://www.srf.ch/audio/kultur-talk)                                                                                                    | 55.57       | 51.33         | v1.0          |
| [Literaturclub - Zwei mit Buch](https://www.srf.ch/audio/literaturclub-zwei-mit-buch)                                                                  | 31.65       | 28.04         | v1.0          |
| [Medientalk](https://www.srf.ch/audio/medientalk)                                                                                                      | 68.77       | 62.16         | v1.0          |
| [Persönlich](https://www.srf.ch/audio/persoenlich)                                                                                                     | 763.15      | 637.87        | v2.0          |
| [Pipifax](https://www.srf.ch/audio/pipifax/eigene-beduerfnisse-wie-nehme-ich-mir-zeit-fuer-mich-selber-1-20?uuid=8c53e199-78dd-4ae9-8337-f6bc08286967) | 9.04        | 7.66          | v1.0          |
| [Podcast am Pistenrand](https://www.srf.ch/audio/podcast-am-pistenrand)                                                                                | 18.16       | 15.37         | v1.0          |
| [Ratgeber](https://www.srf.ch/audio/ratgeber)                                                                                                          | 574.46      | 445.64        | v2.0          |
| [Rehmann](https://www.srf.ch/audio/rehmann)                                                                                                            | 213.87      | 182.79        | v2.0          |
| [Samstagsrundschau](https://www.srf.ch/audio/samstagsrundschau)                                                                                        | 414.45      | 382.33        | v1.0          |
| [Sternstunde Philosophie](https://www.srf.ch/audio/sternstunde-philosophie)                                                                            | 158.67      | 136.70        | v1.0          |
| [Sternstunde Religion](https://www.srf.ch/audio/sternstunde-religion)                                                                                  | 60.58       | 53.90         | v1.0          |
| [Sykora Gisler](https://www.srf.ch/audio/sykora-gisler)                                                                                                | 149.49      | 125.80        | v1.0          |
| [Tagesgespräch](https://www.srf.ch/audio/tagesgespraech)                                                                                               | 1688.26     | 1557.43       | v1.0          |
| [Ufwärmrundi](https://www.srf.ch/audio/ufwaermrundi)                                                                                                   | 60.72       | 54.95         | v1.0          |
| [Vetters Töne](https://www.srf.ch/audio/vetters-toene)                                                                                                 | 25.37       | 20.13         | v1.0          |
| [Wetterfrage](https://www.srf.ch/audio/wetterfrage)                                                                                                    | 65.52       | 59.02         | v1.0          |
| [Wirtschaftswoche](https://www.srf.ch/audio/wirtschaftswoche)                                                                                          | 126.23      | 115.31        | v1.0          |
| [Wissenschaftsmagazin](https://www.srf.ch/audio/wissenschaftsmagazin)                                                                                  | 403.10      | 347.52        | v1.0          |
| [Zivadiliring](https://www.srf.ch/audio/zivadiliring)                                                                                                  | 49.80       | 42.55         | v1.0          |
| [Zytlupe](https://www.srf.ch/audio/zytlupe)                                                                                                            | 45.66       | 36.61         | v1.0          |
| **Total**                                                                                                                                              | **9041.28** | **7765.72**   |               |

| **YouTube Podcast Name**                                                                                                    | **Raw (h)** | **Clean (h)** | **vSwissGPC** |
|-----------------------------------------------------------------------------------------------------------------------------|-------------|---------------|---------------|
| [Auf Bewährung - Leben mit Gefängnis](https://www.youtube.com/playlist?list=PLAD8a6PKLsRhHc-uS6fA6HTDijwE5Uwju)             | 3.00        | 2.70          | v1.0          |
| [Berner Jugendtreff](https://www.youtube.com/playlist?list=PLyWje_91744G6UAsfHjTLWDtejJdHmuYv)                              | 127.80      | 89.61         | v1.0          |
| [Ein Buch Ein Tee](https://www.youtube.com/playlist?list=PLCospSPttrrVSk0N5Mqj1dveKZtDZNOAl)                                | 3.73        | 3.26          | v1.0          |
| [expectations - geplant und ungeplant kinderfrei](https://www.youtube.com/playlist?list=PL5ZbqYujTUkVmNCGMP4e0yFVhY8P5EC73) | 16.84       | 14.80         | v1.0          |
| [Fadegrad](https://www.youtube.com/playlist?list=PL356t1Y2d_AXycvLzBF1n8ee0uM4pw9JX)                                        | 49.95       | 42.40         | v1.0          |
| [Feel Good Podcast](https://www.youtube.com/playlist?list=PLf-k85Nq3_j-glR2im1SZv_BxqzdYdENk)                               | 319.60      | 261.43        | v1.0          |
| [Finanz Fabio](https://www.youtube.com/playlist?list=PLGJjtm2tSyhQXU-_N2YkfqCffXhY6UHNe)                                    | 58.44       | 49.29         | v1.0          |
| [Scho ghört](https://www.youtube.com/playlist?list=PLKaFe_fDMhQNbWvnJGC6HArb285ZUdGbz)                                      | 23.45       | 20.47         | v1.0          |
| [Sexologie - Wissen macht Lust](https://www.youtube.com/playlist?list=PL3D2QP2F5r9VDSj6YQb6Ihr_63Gxtm4L5)                   | 15.41       | 13.57         | v1.0          |
| [SRF Dokumentationen](https://www.youtube.com/playlist?list=PLrAvDZ9sYjXYQb1Jk4TSyy6JTXbDn1JLg)                             | 398.73      | 284.01        | v2.0          |
| [SRF Reportagen](https://www.youtube.com/playlist?list=PLrAvDZ9sYjXZ72dR3c-xdnAjRIVmTIMIG)                                  | 196.39      | 148.10        | v2.0          |
| [Über den Bücherrand](https://www.youtube.com/playlist?list=PLPtjJ0sjI3yzhNtZUBY0_e462_gKtr90V)                             | 14.53       | 12.59         | v1.0          |
| [Ungerwegs Daheim](https://www.youtube.com/playlist?list=PLM4IdPP-Tx3W84w1GB8cn33GnuIGcqaeP)                                | 38.67       | 31.08         | v1.0          |
| [Wir müssen reden - Public Eye spricht Klartext](https://www.youtube.com/playlist?list=PLtTxFB6b5Pljl4RU6vimwfQpV490K6SQe)  | 17.52       | 15.54         | v1.0          |
| **Total**                                                                                                                   | **1277.47** | **988.85**    |               |

## Data pipeline
The data from YouTube is downloaded using [pytubefix](https://github.com/JuanBindez/pytubefix) while the SRF data was sourced via the official [SRF API](https://developer.srgssr.ch/api-catalog). Specifically for YT
the code expects a playlist of videos instead of just a video link. This is so that all episodes can be downloaded at once. SRF podcasts only require the name
without any additional information. The pipeline itself is built to download and transcribe the podcasts sequentially, i.e. one podcast after another. The code can of course be changed by you 
to do every step in batch und should not be too much effort to do so. Controlling the pipeline is done via the [config.yaml](config.yaml), in which you can set what podcast should be downloaded
from which source and which pipeline steps should run. See Table below for more information about the parameters. We utilized hdf5 files
in our setup and as such all data is put into hdf5 files on segmentation. This can be changed to your setup accordingly.

| **Config parameter**      | **Description**                                                                   | **Example value for SRF** | **Example Value for YT**                                                 |
|---------------------------|-----------------------------------------------------------------------------------|---------------------------|--------------------------------------------------------------------------|
| source                    | Defines the source of the podcast (either YT or SRF)                              | "srf"                     | "yt"                                                                     |
| youtube_url               | YouTube link to a **Playlist** containing the podcast episodes                    | ""                        | https://www.youtube.com/playlist?list=PLGJjtm2tSyhQXU-_N2YkfqCffXhY6UHNe |
| podcast_name              | Name of podcast as provided by authors                                            | "Zivadiliring"            | "Finanz Fabio"                                                           |
| write_attrs_to_hdf5       | Should attributes (i.e. annotated data) be added to the hdf5 files                | false                     | false                                                                    |
| steps/download            | False/True: Should download step be executed                                      | true                      | true                                                                     |
| steps/diarization         | False/True: Should diarization step be executed                                   | true                      | true                                                                     |
| steps/segmentation        | False/True: Should segmentation step be executed                                  | true                      | true                                                                     |
| steps/phon_transcription  | False/True: Should phoneme transcription step be executed                         | true                      | true                                                                     |
| steps/ch_transcription    | False/True: Should dialect classification step be executed                        | false                     | false                                                                    |
| steps/mel_spectogram      | False/True: Should mel spectrogram generation step be executed                    | false                     | false                                                                    |
| steps/move_into_dialect_5 | False/True: Should audio be moved from podcast-based hdf5 to unified dialect hdf5 | false                     | false                                                                    |
