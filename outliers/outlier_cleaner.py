#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    # Calculate residuals and create tuples of (age, net_worth, error)
    for i in range(len(predictions)):
        error = abs(predictions[i] - net_worths[i])  # Calculate the error
        cleaned_data.append((ages[i], net_worths[i], error))  # Create a tuple

    # Sort the cleaned_data list by error (the third element of the tuple)
    cleaned_data.sort(key=lambda x: x[2])  # Sort by error

    # Determine how many points to remove (10% of the total)
    num_outliers = int(0.1 * len(cleaned_data))

    # Return the cleaned data without the outliers
    return cleaned_data[:-num_outliers]  # Return all but the largest errors
