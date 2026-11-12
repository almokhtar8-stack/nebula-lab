Run on 12 November 2026 in Google Colab: free runtime, 2 CPU cores, no GPU, Python 3.13.18, NumPy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, Matplotlib 3.10.0. The notebook was opened from the `main` branch through the Colab badge link, and it downloaded `nebula_lab.py` and the data from the `v1.0` branch. Colab's package versions sit between the two local test environments above.

**Result.** Every cell ran without an error in three fresh runtimes, and again after "Restart session and run all". The numbers matched the local runs exactly: 274,275 training examples; the fitted line 0.86 × promoter openness − 0.04; r = 0.62, R² = 0.30, median error 3.3-fold, 70% of genes over-predicted; option B 0.616 against option A 0.621, with weights 0.05 and 0.84; the copy-the-average control 0.77; best-matching training cell type MEP, 0.80. Option A in step ④ gave the "same model, same answer" message.

| Step | Time in Colab, fresh runtime |
|---|---|
| Connect a new runtime and run step ① | 16.5 s from clicking "Run anyway" |
| ① Load the data (download, checksum, read) | 5 to 7.5 s |
| ② Fit the line | 0.8 s |
| ③ Predict and test | 0.5 s |
| ④ Experiment | 0.9 s |
| ⑤ Compare with GET | 0.6 s |
| All ten optional exercises | 14.5 s (every cell type at once 4.2 s, the flexible model 8.5 s) |

**Problems found in Colab and fixed** (commits `a5c76e4` and `aa5c644`, on `main` and `v1.0`):

1. Colab removes inline styles from text cells, so the dark title banner showed as four lines of plain 14 px text. The title now uses Markdown headings.
2. Colab draws an image at the width and height saved with it. When the output area is narrower than the figure, the width shrinks and the height does not, so the figure is squeezed sideways. Every figure was squeezed at the browser pane's width (about 620 px), and the three widest (steps ①, ④ and ⑤, 1,190 to 1,290 px) are also wider than the output area of a typical 1,366-pixel laptop screen. Each figure now carries its width only and scales evenly.
3. Colab keeps each Markdown table cell on one line. The step ⑤ comparison table was 1,404 px wide and made the whole notebook scroll sideways on any narrower window, laptops included. It is now four short rows (about 460 px) with the details in a sentence below.
4. Colab shows a code cell's `#@title` as a heading, so step ② and the ten optional exercises showed the same heading twice. Step ②'s cell is now titled "Fit the line", and the optional cells show their one line of code instead of a title.
5. Printed text never wraps in Colab, so the long sentences under three optional exercises ran off the right edge. They are now wrapped notes.

The start-up hint in the first cell now says "up to half a minute", and the optional exercises give Colab's timings.

**Checked after the fixes:** the whole notebook at the pane's own width (about 620 px), at iPad portrait size (768 × 1024, emulated) and in dark mode. Figures keep their proportions, nothing scrolls sideways, the output cards stay readable on the dark background, the step ④ menu switches between options A and B, and the menu in optional exercise 2 works without a title.

**Phones.** At phone width (375 px, emulated) Colab's own toolbar and output area are cut off at the right edge. That is Colab's layout, not the notebook's. The first cell sends phone users to the project page, which shows every result as phone-sized images.

**Not checked.** A real iPad or phone (the sizes above were emulated in a desktop browser), and whether a KAUST Google account can open Colab. Colab's runtime image changes over time, so run the notebook once more in the week of the session.
