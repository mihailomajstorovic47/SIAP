# This is a sample Python script.
import pandas_csv_service as pd_service
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    #pd_service.merge_tracks_and_artists()
    #pd_service.show_importances()
    pd_service.linear_regression_prediction()
    pd_service.random_forest_prediction()
    #pd_service.show_info()
    #pd_service.plot_graph()
    #pd_service.show_histograms()
    #pd_service.count_very_popular()
    #pd_service.test()