# Football Shootout Analysis

This repository provides tools and methods for analyzing football shootouts. The project focuses on various tasks, including tracking players and the ball, identifying the player with the ball, and dividing teams based on their jerseys using advanced techniques.

## Key Features

- **YOLOv11 for Object Detection**: Leverage the power of YOLOv11 for real-time object detection of players and the ball during shootouts.
- **Team Division by Jersey Color**: Utilize __SigLip__, __UMAP__, and __KMeans__ to cluster players based on their jersey colors, effectively dividing teams for analysis.
- **Player and Ball Tracking**: Implement tracking algorithms to follow players and the ball throughout the shootout.
- **Finding Player with the Ball**: Automatically identify which player has the ball at any given moment.
- 


## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/xAn-x/Football-Shootout-Analysis.git
cd Football-Shootout-Analysis
pip install -r requirements.txt
```

## Original Images vs. Model Results

This section displays the original images alongside the results produced by the model. The comparison highlights the effectiveness of the analysis in detecting and tracking players and the ball.

| Original | Model Result |
|----------|--------------|
| <img src="test/image1.png" alt="Image 1" class="responsive-img"> | <img src="output/image1_output.png" alt="Model Output" class="responsive-img"> |
| <img src="test/image2.png" alt="Image 2" class="responsive-img"> | <img src="output/image2_output.png" alt="Model Output" class="responsive-img"> |
| <img src="test/image2.png" alt="Image 2" class="responsive-img"> | <img src="output/image3_output.png" alt="Model Output" class="responsive-img"> |

### Video

| Original | Model Result |
|----------|--------------|
| <img src="test/video.gif" alt="Video" class="responsive-img" style="width:550px"> | <img src="output/output.gif" alt="Output" class="responsive-img" style="width:550px;"> |

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue if you have suggestions or improvements.
