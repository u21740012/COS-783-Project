from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size= (224,224)
batch_size = 32
train_generator = ImageDataGenerator(rescale=1./255,rotation_range=10,zoom_range=0.1)
validation_generator = ImageDataGenerator(rescale=1./255)
train_data = train_generator.flow_from_directory("data/TRAINING_CG-1050/TRAINING",target_size=image_size,batch_size=batch_size,class_mode='binary')
validation_data = validation_generator.flow_from_directory("data/VALIDATION_CG-1050/VALIDATION",target_size=image_size,batch_size=batch_size,class_mode='binary')
print(train_data.class_indices)
print(validation_data.class_indices)