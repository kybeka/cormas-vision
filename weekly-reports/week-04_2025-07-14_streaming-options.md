## Week 04 – 2025-07-14

### Weekly Focus

**Context:**  
This week focused on experimentation with video input options, model tuning, and continuing infrastructure work for the detection pipeline. Progress was made across multiple tracks, though some setup items are still in progress.

### Tasks

- Explored RTMP streaming between phone and computer, but decided to switch to a wired connection (likely using DroidCam) for reliability and latency reasons.
- Continued fine-tuning models for better precision and robustness.
- Performed ongoing benchmarking and experimentation with different frame sampling rates and confidence thresholds.
- Set up a basic experiment tracker to log model performance and test results.
- Finalized the list of games for visual detection: Chess, Planet-C, and Ubuntu.
- Started organizing logistics for acquiring a tripod and a physical Planet-C game board.
- Continued labeling the ~22k Planet-C frames using Roboflow and manual adjustments via LabelImg.
- Worked on Dockerizing the `chess-v1` pipeline; this includes encapsulating the RTMP connection setup.
- *Still pending:* Creating `roadmap.md` to formalize project milestones.