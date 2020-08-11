import data_service, plotting_service

train_X, train_Y, test_X, test_Y = data_service.load_2D_dataset()
plotting_service.visualize_dataset(train_X, train_Y)