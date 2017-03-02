# bike-sharing-kaggle

Predict bike share demand using two years worth of data for the Kaggle competition.

Strategy used - 

1. Extract day, hour, month and year from the datetime variable.

2. Visualize to look for existing patterns. 

3. Bin temporal variables. 

4. Train two GBM model for registered and casual users.

5. Average out their predictions to make up the total count. 
