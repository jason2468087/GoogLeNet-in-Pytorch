# GoogLeNet in Pytorch

## Introduction
This program is the famous GoogLeNet machine learning CNN model. The code is built with Pytorch library on jupyter notebook. However, due to hardware limitation, the training only lasted for 6 hours and could not continue for higher accuracy.

## Dependencies
- Pytorch 1.7.0
- Python 3.7.4

## Skills
- Inception Structure

## Result

Total test accuracy: 76.43%

Training result visualised:
![alt text](https://github.com/jason2468087/Pytorch-Inception/blob/main/img/Inception%20Result.png?raw=true)

Accuracy per class:
| Airplanes | Automobile | Birds | Cats | Deer | Dogs | Frogs | Horses | Ships | Trucks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 70.48% | 71.91% | 54.81% | 43.14% | 52.22% | 66.19% | 62.76% | 71.10% | 72.91% | 68.57% |

Confusion Matrix:

| Predict\Actual | Airplanes | Automobile | Birds | Cats | Deer | Dogs | Frogs | Horses | Ships | Trucks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Airplanes | 640 | 43 | 73 | 33 | 41 | 3 | 8 | 19 | 98 | 42 |
| Automobile | 49 | 709 | 14 | 35 | 26 | 4 | 7 | 7 | 47 | 102 |
| Birds | 55 | 14 | 541 | 94 | 124 | 25 | 95 | 26 | 16 | 10 |
| Cats | 14 | 18 | 57 | 481 | 88 | 141 | 123 | 31 | 19 | 28 |
| Deer | 14 | 10 | 59 | 56 | 647 | 28 | 84 | 80 | 12 | 10 |
| Dogs | 5 | 13 | 76 | 211 | 57 | 509 | 48 | 56 | 6 | 19 |
| Frogs | 2 | 18 | 54 | 86 | 123 | 22 | 664 | 6 | 9 | 16 |
| Horses | 26 | 10 | 74 | 65 | 94 | 29 | 15 | 625 | 8 | 54 |
| Ships | 80 | 49 | 23 | 32 | 19 | 2 | 10 | 8 | 724 | 53 |
| Trucks | 23 | 102 | 16 | 22 | 20 | 6 | 4 | 21 | 54 | 732 |
