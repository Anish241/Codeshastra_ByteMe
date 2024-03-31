from Modules import prepare,prediction

map = {
    0:"Incorrect",
    1:"Correct"
}

input = prepare(r'.\Testing')
pred = prediction(input)
print(map)
print(pred)