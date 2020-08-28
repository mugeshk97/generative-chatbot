from tkinter import *
import pickle
import numpy as np
from voice import get_audio, put_audio
from util import transform
from model import inference_model

# Tokenizer
tokenizer = pickle.load(open('Asset/Tokenizer/tokenizer.pickle', 'rb'))
# Model
enc_model, dec_model = inference_model()


root = Tk()
root.geometry('500x300')
root.resizable(width=False, height=False) 
root.title('Astro')

put_audio('Astro version 1.0')
def mind(user_input):
	tar = transform(user_input, tokenizer)
	states_values = enc_model.predict(tar)
	empty_target_seq = np.zeros((1, 1))
	empty_target_seq[0, 0] = tokenizer.word_index['start']
	stop_condition = False
	decoded_translation = ''
	while not stop_condition:
	    dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
	    sampled_word_index = np.argmax(dec_outputs[0, -1, :])
	    for word, index in tokenizer.word_index.items():
	        if sampled_word_index == index:
	            decoded_translation += ' {}'.format(word)
	            sampled_word = word
	    if sampled_word == 'end':
	        stop_condition = True
	    empty_target_seq = np.zeros((1, 1))
	    empty_target_seq[0, 0] = sampled_word_index
	    states_values = [h, c]
	out = decoded_translation.split()[:-1]
	decoded_translation = ' '.join(out)
	return decoded_translation


def vocal():
	user_input = get_audio()
	decoded_translation = mind(user_input)
	put_audio(decoded_translation)
	return

def hand():
	user_input = entry.get()
	decoded_translation = mind(user_input)
	return


canvas = Canvas(root,width=500,height=300)
image1 = PhotoImage(file='Asset/Images/bg.png')
canvas.create_image(0,0,anchor=NW,image=image1)
canvas.pack()


photo = PhotoImage(file='Asset/Images/icons8-microphone-25.png')
chat_button = Button(root,image=photo,bg='white',command=vocal)
chat_button.place(x=210,y=130,height=30,width=40)

'''
text_button = Button(root,text='chat',command=hand,height=1,width=3,bg='white')
text_button.place(x=340,y=440,height=30,width=40)
'''
image = PhotoImage(file='Asset/Images/icons8-logout-rounded-left-25.png')
quit_button = Button(root, text = 'Quit',command =root.destroy,image=image,height=1,width=3,bg='white') 
quit_button.place(x=260,y=130,height=30,width=40)
root.mainloop()



