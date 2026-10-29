# 2 to 3 minute demonstration: what to show and say

Record your screen with your voice. On Windows press **Windows + Alt + R** (Game Bar) or use Clipchamp or OBS. Upload the video to YouTube (Unlisted) or Google Drive and put the link in the README. GitHub does not accept large video files.

Prepare first: the app running (`python app.py`) in the browser, 5 test photos on your desktop (3 clear waste items, 1 blurry or unusual item, 1 thing that is not waste such as a banana peel or a pen), and `evaluation_report.md` open in VS Code.

| Time | Show | Say (use your own words) |
|---|---|---|
| 0:00 to 0:15 | README title | "This is my waste image classification system. It sorts a photo into cardboard, glass, metal, paper or plastic." |
| 0:15 to 0:45 | Dataset section / class distribution picture | "I used the TrashNet dataset. I removed exact duplicates and kept near-identical photos together in one split, so no copy of a training image is in the test set. Split sizes are ..." |
| 0:45 to 1:15 | Comparison table and training curves | "I trained a baseline CNN, an improved CNN with augmentation and batch normalisation, and a pretrained MobileNetV2. The test accuracies are ..., ... and .... The best one is my final model." |
| 1:15 to 2:00 | The app: upload 2 or 3 clear photos | "I upload a photo. It gets the same preprocessing as in training. The app shows the category and the score, and the predictions of all three models." |
| 2:00 to 2:30 | Upload the unusual item and the non-waste item | "Here the model is unsure or wrong. Notice it still picks a category, sometimes with a high score. Model scores do not guarantee correctness." |
| 2:30 to 2:50 | Confusion matrix and failure examples | "Most mistakes are between ... and .... To improve I would add more varied photos and an 'unknown' category." |

Keep it under 3 minutes. Do one practice run first.
