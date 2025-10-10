# Screen-cam

## Project Overview
Back illumination screen is controlled by feedback from a camera facing the screen.

## Setup
- Screen lays flat on the table.
- Camera is placed above the screen.
- Samples are placed on the screen, or on a glass surface above the screen.

## Camera-to-screen register
We use a circle grid pattern to register the screen positions of each pixel on the camera. 
Then we have a mapping from camera coordinates to screen coordinates.

## Applications

### Live shadow simulator 
Objects betwene the screen and the camera are detected in the camera image.
The mask of these objects is then mapped to the screen, creating am illusion of a shadow on the screen.

Run:
[shadow.py](shadow.py)

### Darkfield illumination

The backlight screen is illuminated and imaged with moving stripes. Each pixel is averaged over images where it is in a dark stripe. This technique thereby simulates darkfield illumination, where the signal in each pixel is proportional to scattering, not transmission. 

Run:
[dark_field.py](dark_field.py)


