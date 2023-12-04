# thesisF2023
This is the programming part of my DSS thesis. F2023. Here you find the classifier that I trained on my data to predict political leaning based on posts. 
Next to this, you'll find the code for two obfuscation methods (word-to-number mapping and paraphrasing with text-to-text transfer transformer). 
After applying these obfuscation methods, the classifier is used to predict political leaning again, but now based on the obfuscated posts. The performance 
of the classifier tells us something about the preservetion of the privacy of the authors. When the classifier has more trouble predicting the politcal leaning 
of the authors after obfsucation, privacy is preserved. 
