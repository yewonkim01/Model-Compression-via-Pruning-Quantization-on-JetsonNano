import torch.nn as nn
import torch.optim as optim

def re_initialize_weights(model, method):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if method == 'random':
                nn.init.uniform_(layer.weight, -0.1, 0.1)
                nn.init.uniform_(layer.bias, -0.1, 0.1)

            elif method == 'he':
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)




def fine_tuning(model, device, data, lr, finetuning_epoch, re_initialize):

    ######### For checking initialization for finetuning ##########
    if re_initialize == 'random': # random
        re_initialize_weights(model, 'random')

    elif re_initialize == 'he': #He
        re_initialize_weights(model, 'he')

    else: # pruned parameters
        pass

    ###############################################################


    epochs = finetuning_epoch
    learning_rate = lr # smaller than 1e-3(original training lr)
    total_samples = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        total_loss = 0

        for idx, (images, labels) in enumerate(data):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_samples += labels.size(0)
            total_loss += loss.item()

            # if idx % 100 == 0:
            #     print(f"idx {idx}, Runnning loss: {total_loss / len(train_loader)}")
        # print(f"epoch {e} : total_loss : {total_loss}")

    

    return model
