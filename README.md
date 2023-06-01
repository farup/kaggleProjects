# Titanic 

Exploring how to set up pipelines for streamlining and automating the machine learning workflow.
Combines multiple data preprocessing steps and estimator into a single object. Notebook inspired by https://www.youtube.com/@NeuralNine

Brief overview: 
- Step 1. Exploring data
- Step 2. Splitting the data
- Step 3. Setting up data preprocessing 
    -  AgeImputer() : Filling in missing values for 'Age' column
    -   FeatureEncoder() : OneHot encodes categorical data
    -   FeatureDropper() : Drops not useful columns 
    -  Pipeline combines mentioned object. 
-  Step 4. Train with RandomForesFlassifier 
-  Step 5. Test
