# Predicting Extraterrestrial Growth with ML 👽
![Pandas](https://img.shields.io/badge/Pandas-006400?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-003366?style=flat-square&logo=matplotlib&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3F8EFC?style=flat-square&logo=xgboost&logoColor=white)
![Gradient Boosting](https://img.shields.io/badge/Gradient%20Boosting-27AE60?style=flat-square&logo=python&logoColor=white)

## Collaborators: Benedict Neo, Trevor Eaton

**Kaggle Competition: [Predicting Extraterrestrial Growth](https://www.kaggle.com/competitions/adv-ml-2025/overview)**

As scientists in a galaxy far, far away, our research interests lie in understanding the unique growth trajectories of an undisclosed planet’s inhabitants. While these aliens exhibit remarkable similarities to human biology and growth patterns, the field of extraterrestrial growth development remains largely uncharted territory. 

Our research aims to explore this exciting frontier, deepening our understanding of these aliens’ growth trajectories, and ultimately predicting how the aliens develop from age 10 to adulthood (age 18). As alien data scientists, we saw this research problem in the context of machine learning and longitudinal data; hence, we aimed to leverage the predictive power of regression models to tackle this task. 

Throughout our research, we were met with data challenges, including erratic measurements and missing data points, all the while considering the need to model hereditary influences, individual growth patterns, and biological factors. Our final weighted ensemble model leverages insights from exploratory analysis and biologically informed feature selection to accurately capture these complex growth dynamics.

## Building the Models

Our approach to predicting growth trajectories evolved through several stages, guided by cross-validation performance and biological reasoning. We continued to refine our model architecture to capture the complex patterns of growth while maintaining interpretability and robustness.

### Baseline Models
We began with three standard regression algorithms as baseline approaches: Lasso Regression (L1 regularization), Ridge Regression (L2 regularization), and Random Forest. Initial cross-validation tests showed Lasso outperforming Ridge regression (7.284 RMSE vs. 7.408 RMSE). Random Forest achieved a competitive 7.580 RMSE.
Recognizing the complementary strengths of different algorithms, we developed a weighted ensemble approach that combines Lasso and Random Forest learners. Initial cross-validation tests showed Lasso excelled at younger ages (10-13) with RMSE between 1.38-4.71, while Random Forest performed better at older ages (14-18) with RMSE between 7.04-7.90 (relative improvement to Lasso’s 7.34-8.78 RMSE).
This pattern aligns with biological understanding: linear models perform well for shorter-term predictions during relatively stable growth phases, while tree-based models capture the non-linear patterns that emerge during the variable timing of pubertal growth spurts.

### Hyperparameter Tuning
We implemented comprehensive hyperparameter tuning for each algorithm to maximize performance. For Lasso, we optimized the regularization strength (alpha), finding that optimal values varied by prediction age. Younger ages generally required less regularization (lower alpha values), while older ages benefited from stronger regularization to prevent overfitting.
Grid search optimization was utilized for Random Forest, focusing on tree depth, minimum samples for splitting, and ensemble size. Deeper trees (depth 12-15) and larger ensembles (100-150 trees) typically performed best, with age-specific variations. This age-specific tuning significantly improved our model performance, especially for earlier ages:
| Age | 10  | 11  | 12  | 13  | 14  | 15  | 16  | 18  |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| RMSE | 1.28 | 2.26 | 3.30 | 4.57 | 5.88 | 6.62 | 7.07 | 7.60 |

The increasing RMSE with prediction age reflects the inherent challenge of biological data—growth simply becomes more variable and harder to predict during and after puberty.

Expanding the Ensemble with Gradient Boosting
To further enhance model performance, we expanded our ensemble to include Gradient Boosting Regression. While Random Forest builds an ensemble of independent trees, Gradient Boosting builds shallower trees sequentially to correct previous errors, potentially capturing different growth patterns. Additionally, Gradient Boosting performs well in modeling complex non-linear relationships, which seemed fitting for growth especially during pubertal transitions. Finally, our initial testing showed Gradient Boosting performed particularly well at transitional ages (11 and 14), where growth patterns shift from childhood to early puberty and from early to late puberty depending on the sex.
We tuned this algorithm based on learning rate, tree depth, and number of estimators, finding that modest learning rates (0.05-0.1) with moderate tree depths (3) provided the best balance of performance and generalization.
The inclusion of the tuned Gradient Boosting learner improved our cross-validation performance, particularly at these transition ages, with the final ensemble outperforming any two-model combination. 

### Weight Optimization
Another key contributor to performance was the use of validation-based weight optimization for combining predictions. Rather than using fixed weights or simple averaging, we implemented a constrained optimization approach to find the optimal contribution of each algorithm. The optimized weights revealed patterns about algorithm performance across ages:
| Age | Lasso Weight | RF Weight | GB Weight |
|-----|--------------|-----------|-----------|
| 10  | 0.734        | 0.266     | 0.000     |
| 11  | 0.649        | 0.215     | 0.136     |
| 12  | 0.763        | 0.237     | 0.000     |
| 13  | 0.908        | 0.092     | 0.000     |
| 14  | 0.521        | 0.324     | 0.155     |
| 15  | 0.476        | 0.524     | 0.000     |
| 16  | 0.477        | 0.523     | 0.000     |
| 18  | 0.423        | 0.577     | 0.000     |

### Ensemble Design
Our final ensemble architecture incorporated three algorithms, each contributing distinct strengths to growth prediction: 
1.	Lasso Regression: Provides interpretability through feature selection and performs well in situations with limited data relative to features. Lasso excels at capturing linear growth patterns during stable growth phases, which it demonstrated for short-term predictions (ages 10-13).
2.	Random Forest: Captures non-linear relationships and handles interactions between features. It performed quite well for longer-term predictions (ages 15-18)
3.	Gradient Boosting: Focuses on difficult-to-predict cases through sequential error correction, contributes most significantly at ages 11 and 14, coinciding with typical transition points.

### Age-Specific Feature Analysis
We trained separate models for each target age rather than a single model predicting all ages based on insights from our model development process in which we noticed that feature importance varies not only by algorithm but also by prediction age.

#### Early Ages (10-12)
- For Lasso at age 10, `early_max_height` dominates with minimal contribution from other features. By age 12, while `early_max_height` remains important, `recent_growth_rate` begins gaining significance.
- Random Forest shows a more balanced distribution even at early ages, with `early_max_height` and `recent_growth_rate` sharing importance.

#### Middle Ages (13-15)
- Random Forest at age 13-15, `recent_growth_rate` begins to overtake `early_max_height`.
- Gradient Boosting shows a dramatic shift at age 14, with `recent_growth_rate` and `early_acceleration` becoming dominant features.

#### Later Ages (16-18)
- Random Forest: `recent_growth_rate` and `early_acceleration` are the two most important features.
- Lasso maintains emphasis on `early_max_height` but with increased weight on `recent_growth_rate`.

This evolving pattern of feature importance provides strong biological justification for our age-specific modeling approach. For ages 10-12, early maximum height (a proxy for genetic growth potential) is the strongest predictor for future growth trajectory, indicating that hereditary factors play a dominant role in early growth. For ages 13-15, when many children enter puberty, recent growth rate becomes increasingly important as it signals the onset and intensity of the pubertal growth spurt. Finally, for ages 16-18, acceleration-related features gain importance as they help identify where subjects are in their growth trajectory (accelerating, at peak velocity, or decelerating). Differentiating by sex also becomes important in these later ages due to the unique growth patterns between males and females during and after puberty.
Training age-specific models allowed us to capture these shifting biological determinants of growth across development, resulting in better performance than a single model could achieve. Each model effectively specialized in the unique growth dynamics of a specific developmental stage, with optimized weightings of our ensemble components specializing them further.

### Final Ensemble Performance
| Age | 10   | 11   | 12   | 13   | 14   | 15   | 16   | 18   |
|-----|------|------|------|------|------|------|------|------|
| RMSE| 1.25 | 2.25 | 3.26 | 4.57 | 5.88 | 6.56 | 7.07 | 7.57 |

### Model Limitations
Despite the good performance of our final model, we wanted to acknowledge limitations associated with it. Firstly, our ensemble approach, while effective for prediction, introduces complexity that makes interpretation more challenging. This trade-off between accuracy and interpretability could be a consideration for researchers primarily interested in understanding growth patterns and factors rather than prediction alone.
Our analysis was also constrained by the dataset. The relatively small sample size restricted our ability to create robust validation sets while maintaining sufficient training data, potentially affecting our model's generalizability to the broader alien population. Additionally, for approximately 15% of subjects with missing parent data, we relied on imputation techniques that may not fully capture complex hereditary influences. We also missed opportunities to leverage potentially valuable features—most notably, we did not fully exploit the relationship between weight and height measurements, which could have provided additional predictive power.
Looking forward, several approaches could enhance performance in future iterations. Deep learning methods like neural networks might better capture the sequential nature of growth patterns. More sophisticated imputation methods like Gaussian processes could provide better estimates for missing measurements, and incorporating additional biological markers beyond height could yield more accurate and biologically meaningful predictions.

## Conclusion
Our final model achieved a public Kaggle score of 12.5204, made possible from several key insights that emerged during our development process. Firstly, our biologically informed feature engineering approach captured the dynamics of growth in this alien race, focusing on growth potential (early maximum height), growth momentum (recent growth rate), and sex-specific patterns. These features aligned with established alien growth literature and proved consistently important across our models.
Secondly, our weighted ensemble architecture leveraged the complementary strengths of different algorithms. Lasso excelled at early-age predictions when growth patterns were more linear, while Random Forest captured the complex patterns at later ages. Gradient Boosting provided contributions at critical transition points (ages 11 and 14), helping to better predict the non-linear nature of growth during these periods.
Thirdly, the age-specific modeling approach recognized that different biological factors drive growth at different developmental stages. The shifting patterns of feature importance across ages confirmed that a one-size-fits-all model would have failed to capture the evolving determinants of growth through adolescence.
Finally, our data cleaning and preprocessing, including parent trajectory reconciliation and linear interpolation of missing values, ensured we preserved the biological signal while minimizing noise.
We are excited to work with more alien biologists in the future to refine our model and make our research more applicable to the entire alien race. To infinity and beyond!
