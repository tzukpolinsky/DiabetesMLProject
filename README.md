```markdown
# train_using_optuna.py

This script uses the Optuna library as part of school project for hyperparameter optimization of various classifiers using Optuna.
It optimizes the classifier configurations on a diabetes 130-US hospitals dataset using LightGBM, xgboost ,BalancedBaggingClassifier, BalancedRandomForestClassifier, RUSBoostClassifier, \
EasyEnsembleClassifier and other known classifiers.
The optimization process focuses on improving validation accuracy.

## Goal
Our goal in this project was to check if we could identify patients that would returned again under 30 days. 
This is could be a major money saver for the hospitals, because in most cases from the literature we reviewed alot of those readdmisions could be prevneted using phone call follow-up.
Unfortunately the data in the UCI file 'Diabetes 130-US hospitals for years 1999-2008' didn't contained enough variance in the categorisation of the patients and therefor there is too much overlapping between classes

### Usage

1. Install the required libraries using the following command:

   ```bash
   pip install joblib matplotlib seaborn pandas numpy scikit-learn optuna lightgbm imbalanced-learn xgboost 
   ```

2. Run the script with the appropriate command-line arguments:

   ```bash
   python train_using_optuna.py [path_to_diabetic_data.csv_file] [result_output_directory] [number_of_trials] [use_smote] [smote_index] [use_filter]
   ```

   - `path_to_data_file`: Path to the diabetic_data.csv data file.
   - `result_output_directory`: Directory where the optimization results and plots will be saved.
   - `number_of_trials`: Number of optimization trials to perform.
   - `use_smote`: Whether to use Synthetic Minority Over-sampling Technique (SMOTE).
   - `smote_index`: Index indicating the type of SMOTE technique to use.
   - `use_filter`: Whether to use our own data filtering to aggressively remove any identical between classes.

3. The script optimizes various classifiers' hyperparameters and saves the results, including classification reports, ROC curves, and optimization history plots, in the specified output directory.

## File Structure

- `train_using_optuna.py`: The main script containing hyperparameter optimization logic.
- `utils.py`: Utility functions used by the main script.
- `README.md`: This documentation file.

## Dependencies

- Python 3.6+
- `optuna`
- `lightgbm`
- `imbalanced-learn`
- `xgboost`

## References

- [diabetic data file](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- [Optuna Documentation](https://optuna.readthedocs.io/en/stable/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)

## License

This project is licensed under the [MIT License](LICENSE).

```