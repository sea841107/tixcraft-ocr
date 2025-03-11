# Tixcraft-OCR

## Concept
1. Design a program to automatically generate captchas. (*image_generator.py*)
2. Use [ddddocr](https://github.com/sml2h3/ddddocr) to recognize captchas and collect.
3. Use the captchas by steps 1 & 2 to train a model.
4. Use the new model to recognize and collect.
5. Use the captchas recognized by new model to train a better model.
6. Repeat steps 4 & 5

### Version 1
```
Total collected: 20000
Correct: 11297
Wrong: 8703
Accuracy: 56.485%
```

### Version 2
```
Total collected: 10000
Correct: 9871
Wrong: 129
Accuracy: 98.71%
```