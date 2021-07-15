import numpy as np
import miditoolkit
from miditoolkit.midi import parser as mid_parser
import pickle
from glob import glob
import copy
from datetime import datetime
import pandas as pd

#############################################################################################
# DATA TRANSFORMATION: MIDI TO INPUT SEQUENCE
#############################################################################################

# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch)

# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path):
    # Read MIDI (example)
    midi_obj = mid_parser.MidiFile(file_path)
    # note
    note_items = []
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    for note in notes:
        note_items.append(Item(
            name='Note', 
            start=note.start, 
            end=note.end, 
            velocity=note.velocity, 
            pitch=note.pitch))
    note_items.sort(key=lambda x: x.start)
    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='Tempo',
            start=tempo.time,
            end=None,
            velocity=None,
            pitch=int(tempo.tempo)))
    tempo_items.sort(key=lambda x: x.start)
    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=existing_ticks[tick]))
        else:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch))
    tempo_items = output
    return note_items, tempo_items

# quantize items (quantize 'start' value of each items to 120*n)
def quantize_items(items, ticks=120):
    # grid
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    #print("grids of quantize items in utils:", grids)
    # process
    for item in items:
	# find the nearest grid index
        index = np.argmin(abs(grids - item.start))
	# calculate the difference between grid and event start time
        shift = grids[index] - item.start
	# shift the whole item to the grid
	# (though the event end time is not on the grid time)
        item.start += shift
        item.end += shift
    return items

# group items (after quantize_items)
def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION*4):
    # because DEFAULT_RESOLUTION*4 = 480 * 4 = 120 * 16, 
    # there are 16 possipable positions inside each group
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups

# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)

# Item to Event
def item2event(groups):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        # skip those empity groups
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        # <bar> event
        events.append(Event(
            name='Bar',
            time=None, 
            value=None,
            text='{}'.format(n_downbeat)))
        for item in groups[i][1:-1]:
            # <position> event
            # flag the start of 16 positions
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            # put the item at the closest position
            index = np.argmin(abs(flags-item.start))
            events.append(Event(
                name='Position', 
                time=item.start,
                value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                text='{}'.format(item.start)))
            if item.name == 'Note':
                # velocity
                velocity_index = np.searchsorted(
                    DEFAULT_VELOCITY_BINS, 
                    item.velocity, 
                    side='right') - 1
                events.append(Event(
                    name='Note Velocity',
                    time=item.start, 
                    value=velocity_index,
                    text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
                # pitch
                events.append(Event(
                    name='Note On',
                    time=item.start, 
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                # duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                events.append(Event(
                    name='Note Duration',
                    time=item.start,
                    value=index,
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
            # elif item.name == 'Tempo':
                # tempo = item.pitch
                # tempo_value = Event('Tempo Value', item.start, tempo, None)
                # events.append(tempo_value)     
    return events

# MIDI to Words
def mid_to_words(file_list=None, path=None, dictionary_path='dictionary_B.pkl'):
    if file_list==None: # it is a file path contained with songs
        d_rout = glob('{}/*.mid'.format(path))
    else: d_rout = file_list # it is a list of songs name path

    all_items, all_events, all_words = [], [], []

    # MIDI files to Words, song by song
    for d in d_rout:
        # MIDI file to Items (MIDI Note events)
        # 1) Read MIDI file (example)
        midi_obj = mid_parser.MidiFile(d)
        # 2) MIDI file to items (MIDI Note events)
        note_items, tempo_items = read_items(d)
        # 3) Quantize time values of Items from real number (ms) to multiples of 120 (ms)
        note_items = quantize_items(note_items)
        # 4) Group items
        items = note_items # + tempo_items
        # 5) Save Items (MIDI Note events)
        all_items.append(items)

        # MIDI file to REMI Events
        max_time = note_items[-1].end
        # 1) Group events into bars (120*16 ms per bar) -> 16 positions
        groups = group_items(items, max_time)
        # 2) Write Events according to Items
        events = item2event(groups)
        # 3) Save REMI Events
        all_events.append(events)
        
        # REMI events to Words
        event2word, word2event = pickle.load(open(dictionary_path, 'rb'))
        words = []
        # 1) Transform REMI events to Words by dictionary
        for event in events:
            e = '{}_{}'.format(event.name, event.value)
            if e in event2word:
                words.append(event2word[e])
            else:
                # 2) out of values
                if event.name == 'Note Velocity':
                    # 2.1) replace with max velocity
                    words.append(event2word['Note Velocity_32'])
                else:
                    # 2.2) printout "something is wrong"
                    print('something is wrong! {}'.format(e))
        # 3) Save Words
        all_words.append(words)

    return all_words, all_events, all_items

# erase the truncation (note)
def erase(data, lower = 1, upper = 176, seq_len=180, erase_value=0, count=False):
    # data: data
    # lower, upper: range of value that should be erase; default lower = 1, upper = 176 (not a duration event)
    # seq_len: default = 180
    # erase_value: default = 0
    count = 1
    if count:
        while ((lower<=data[seq_len-count])&(data[seq_len-count]<=upper)):
            count = count + 1
        count-=1
        if count==0: return data
        return data[:-count]
    while ((lower<=data[seq_len-count])&(data[seq_len-count]<=upper)):
        # the last event of x is position (1 - 16) or velocity (17 - 48) or note on (49 - 176)
        data[seq_len-count] = erase_value
        count = count + 1
    return data

# cut the long bar or fill the short bar
def cutorfill(data, seq_len=180, erase_value=0):
    data = data.tolist()
    if max(len(data), seq_len)==len(data):
        return erase(data = data[:seq_len])
    fill_list = [0]*(seq_len-len(data))
    return erase(data=data+fill_list)

# Words to Pair
def words_to_pairs(seq_len=180, path=None, all_words=None):
    all_pairs = [] # list (total_pair_number*2*45*4)
    all_pair_song = [] # list (bar_num*180*1)
    # all_raw_pair_song = [] # list (bar_num*180*1)

    # load data
    if all_words==None: all_words = pickle.load(open(path, 'rb'))
    bar_details = pd.DataFrame({})
    
    for i, words in enumerate(all_words):
        words = np.copy(np.array(words))
        # collect the index of bar event (bar_index) 
        # and the total events num inside the bar (bar_gap)
        bar_index = [i for i, e in enumerate(words) if e == 0]
        gap_end = np.copy(bar_index[1:])
        gap_start = np.copy(bar_index[:-1])
        bar_gap = gap_end - gap_start
        bar_num = len(bar_gap)
        # debug: bar
        event_num = np.array(bar_gap)-1
        empity_num = (seq_len-event_num).tolist()
        df = pd.DataFrame({'song_index': [i+1]*len(gap_start), 'gap_start': gap_start, 'gap_end': gap_end, 
                           'bar_gap': bar_gap, 'event_num': event_num, 'empity_num': empity_num})
        bar_details = bar_details.append(df)
        # start paring (x, y) according to the assigned paring strategy 9 (bar by bar)
        pair_song, raw_pair_song = [], []
        for i in range(0, bar_num, 1): # i: the index of bar event happened
            temp_pair_song = cutorfill(data = words[bar_index[i]+1:bar_index[i]+bar_gap[i]], 
                                       seq_len=seq_len, 
                                       erase_value=0) # list (180*1)
            # reshape temp_pairs to ?*4
            temp_pair_song = np.reshape(temp_pair_song, (-1, 4), order='C') 
            raw_pair_song.append(temp_pair_song.tolist())
            # initialize the value according to its column (cancel the range strategy)
            temp_pair_song -= np.array([0, 16, 48, 176]) 
            # correct those negative value to 0 value
            for i, pair in enumerate(temp_pair_song):
                if (pair==np.array([0, -16, -48, -176])).all(): temp_pair_song[i]*=0
            pair_song.append(temp_pair_song.tolist())
        for i in range(len(pair_song)-1):
            x = pair_song[i]
            y = pair_song[i+1]
            all_pairs.append([x, y])
        # all_raw_pair_song.append(raw_pair_song)
        all_pair_song.append(pair_song)
    bar_details.to_csv('data_visualization/ave_note_per_song_before_{}.csv'.format(datetime.now().strftime("%m%d%H%M")))

    print("> len(all_pair_song)", len(all_pair_song))
    print("> all_pairs.shape:", np.array(all_pairs).shape)
    return all_pair_song, all_pairs

#############################################################################################
# STATISTIC DISCRIPTION
#############################################################################################

# data for plot
def stat_raw(data):
    data_counted = np.reshape(data, (-1, 4), order='C')
    data_p = np.reshape(data_counted[:, 0], (-1))
    data_v = np.reshape(data_counted[:, 1], (-1))
    data_n = np.reshape(data_counted[:, 2], (-1))
    data_d = np.reshape(data_counted[:, 3], (-1))
    df = pd.DataFrame({'position': data_p, 
                        'velocity': data_v, 
                        'pitch': data_n, 
                        'duration': data_d})
    return df

# counting results for chart
def stat_counting(data):
    data_counted = np.reshape(data, (-1, 4), order='C')
    note_num = data_counted.shape[0]
    data_p = np.reshape(data_counted[:, 0], (-1))
    data_v = np.reshape(data_counted[:, 1], (-1))
    data_n = np.reshape(data_counted[:, 2], (-1))
    data_d = np.reshape(data_counted[:, 3], (-1))
    # position
    ave_position = sum(data_p)/note_num
    std_position = np.std(data_p)
    # velocity
    ave_velovity = sum(data_v)/note_num
    std_velocity = np.std(data_v)
    # note on (pitch)
    (unique, counts) = np.unique(data_n, return_counts=True)
    unique_pitch_num = len(unique)
    max_repeated_pitch_count = max(counts)
    ave_pitch = sum(data_n)/note_num
    std_pitch = np.std(data_n)
    # duration
    ave_duration = sum(data_d)/note_num
    std_duration = np.std(data_d)

    df = pd.DataFrame({'ave_position': [ave_position], 'std_position': std_position, 
                       'ave_velovity': ave_velovity, 'std_velocity': std_velocity,
                       'ave_pitch': ave_pitch, 'std_pitch': std_pitch, 'unique_pitch_num': unique_pitch_num, 'max_repeated_pitch_count': max_repeated_pitch_count, 
                       'ave_duration': ave_duration, 'std_duration': std_duration})
    
    return df

# data for stat
def stat_data(seq_len=180, path=None, all_words=None, usage='bar'):
    # load data
    if all_words==None: all_words = pickle.load(open(path, 'rb'))
    all_bar_dataset = list([])
    all_bar_song_dataset = list([])
    song_df = pd.DataFrame({})
    for words in all_words:
        words = np.copy(np.array(words))
        # collect the index of bar event (bar_index) 
        # and the total events num inside the bar (bar_gap)
        bar_index = [i for i, e in enumerate(words) if e == 0]
        gap_end = np.copy(bar_index[1:])
        gap_start = np.copy(bar_index[:-1])
        bar_gap = gap_end - gap_start
        bar_num = len(bar_gap)
        df = pd.DataFrame({'bar_num': [bar_num], 'ave_note_num': sum((bar_gap-1)/4)/bar_num, 'std_note_num': np.std((bar_gap-1)/4)})
        song_df = song_df.append(df)
        # start paring (x, y) according to the assigned paring strategy 9 (bar by bar)
        pair_song = []
        for j in range(0, bar_num, 1): # i: the index of bar event happened
            if j==0: 
                temp_pair_song = words[bar_index[j]+3:bar_index[j]+bar_gap[j]]
            else:
                temp_pair_song = words[bar_index[j]+1:bar_index[j]+bar_gap[j]]
            if max(len(temp_pair_song), seq_len)==len(temp_pair_song):
                temp_pair_song = temp_pair_song[:seq_len]
                temp_pair_song = erase(temp_pair_song, count=True)
                # debug
                if len(temp_pair_song)==0: 
                    print('erase(count=True) has bug!')
                    break
            # reshape temp_pairs to ?*4
            temp_pair_song = np.reshape(temp_pair_song, (-1, 4), order='C') 
            # initialize the value according to its column (cancel the range strategy)
            temp_pair_song -= np.array([0, 16, 48, 176]) 
            # statistice values inside every bar
            temp_pair_song = np.reshape(temp_pair_song, (-1)).tolist()
            # debug
            if len(temp_pair_song)==0: 
                print('stat_data() modify process has bug!')
                break
            all_bar_dataset.append(temp_pair_song)
            pair_song = pair_song + temp_pair_song


        # statistice values inside every song
        all_bar_song_dataset.append(pair_song)
    if usage=='stat':
        return song_df
    if usage=='song': 
        return all_bar_song_dataset
    return all_bar_dataset
    

#############################################################################################
# WRITE MIDI: OUTPUT SEQUENCE BACK TO MIDI
#############################################################################################

def predictions_to_words(predictions):
    words = []
    # Calculation:
    # back to dictionart.pkl's value (prediction's size: sequence_num*45*4)
    output = predictions + np.array([0, 16, 48, 176])
    # Insert & remove elements:
    for sequence in output:
        # sequence's size: 45*4
        # 1) remove those empty events
        sequence = sequence[sequence!=np.array([0, 16, 48, 176])]
        # 2) insert bar event (0) at the head of prediction
        sequence = np.insert(sequence, 0, 0)
        words = words+sequence.tolist()
        
    # insert position (1) and tempo event (331 = '120'-30+241)
    words.insert(1, 331)
    words.insert(1, 1)
    # size: ?*1, %4=0
    return words

def words_to_midi(words, midi_name='result/test_{}.mid'.format(datetime.now().strftime("%m%d%H%M")), dictionary_path='dictionary_B.pkl'):
    # word to midi
    event2word, word2event = pickle.load(open(dictionary_path, 'rb'))
    output_path = midi_name
    write_midi(words=words, 
               output_path=output_path)

# Transform Words (size: ?*1) back to Events
def words_to_events(words, dictionary_path='dictionary_B.pkl'):
    # word to REMI event
    event2word, word2event = pickle.load(open(dictionary_path, 'rb'))
    events = []
    # word (size: 1) by word
    for word in words:
        event_name, event_value = word2event.get(word).split('_')
        events.append(Event(event_name, None, event_value, None))
    return events

def write_midi(words, output_path, prompt_path=None):
    events = words_to_events(words=words)
    # get downbeat and note (no time)
    temp_notes = []
    temp_tempos = []
    for i in range(len(events)-3):
        if events[i].name == 'Bar' and i > 0:
            temp_notes.append('Bar')
            temp_tempos.append('Bar')
        elif events[i].name == 'Position' and \
            events[i+1].name == 'Note Velocity' and \
            events[i+2].name == 'Note On' and \
            events[i+3].name == 'Note Duration':
            # start time and end time from position
            position = int(events[i].value.split('/')[0]) - 1
            # velocity
            index = int(events[i+1].value)
            velocity = int(DEFAULT_VELOCITY_BINS[index])
            # pitch
            pitch = int(events[i+2].value)
            # duration
            index = int(events[i+3].value)
            duration = DEFAULT_DURATION_BINS[index]
            # adding
            temp_notes.append([position, velocity, pitch, duration])

        elif events[i].name == 'Position' and \
            events[i+2].name == 'Tempo Value':
            position = int(events[i].value.split('/')[0]) - 1
            tempo = int(events[i+1].value)+30
            temp_tempos.append([position, tempo])
            
    # get specific time for notes
    ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = DEFAULT_RESOLUTION * 4 # assume 4/4
    notes = []
    current_bar = 0
    for note in temp_notes:
        if note == 'Bar':
            current_bar += 1
        else:
            position, velocity, pitch, duration = note
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            # duration (end time)
            et = st + duration
            notes.append(miditoolkit.Note(velocity, pitch, st, et))
            
    # get specific time for tempos
    tempos = []
    current_bar = 0
    for tempo in temp_tempos:
        if tempo == 'Bar':
            current_bar += 1
        else:
            position, value = tempo
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            tempos.append([int(st), value])
        
    midi = miditoolkit.midi.parser.MidiFile()
    midi.ticks_per_beat = DEFAULT_RESOLUTION
    # write instrument
    inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
    inst.notes = notes
    midi.instruments.append(inst)
    # write tempo
    tempo_changes = []
    for st, bpm in tempos:
        tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
    midi.tempo_changes = tempo_changes
        
    # write
    midi.dump(output_path)
