import os
import numpy as np
import pandas as pd
import swifter
from typing import Union
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def main(time_block: Union[float, int] = 10, sample_freq: Union[float, int] = 16):
    """

    :param time_block:
    :param sample_freq:
    :return:
    """
    # get all pids (correspond to each subject)
    pids = np.loadtxt('./data/pids.txt', dtype='str')
    for p in pids:

        df = pd.read_csv(f'./data/clean_tac/{p}_clean_TAC.csv')
        df['timestamp'] = pd.to_datetime(df.timestamp, unit='s')

        acc = pd.read_csv('./data/all_accelerometer_data_pids_13.csv',
                          engine='c',
                          dtype={'time': np.int64, 'pid': str, 'x': float, 'y': float, 'z': float}
                          )

        # grab an example pid
        acc = acc[acc.pid == p]
        acc['time'] = pd.to_datetime(acc.time, unit='ms')
        acc.drop_duplicates(subset='time', inplace=True)
        acc['sec_group'] = pd.to_datetime(acc.time.dt.strftime('%Y-%m-%d %H:%M:%S'))
        disqual = acc.sec_group.value_counts()[acc.sec_group.value_counts() < sample_freq].index
        acc = acc[~acc.sec_group.isin(disqual)]

        # resample down to desired hz using mean
        freq_conv = f"{round(1 / sample_freq, 5)}S"
        acc.set_index('time', inplace=True)
        acc = acc.groupby(['sec_group'])[['x', 'y', 'z']].resample(freq_conv).mean().reset_index()

        # remove any nans
        acc.dropna(how='any', inplace=True)
        start = acc.sec_group.min() - pd.to_timedelta(1, 's')
        end = acc.sec_group.max() + pd.to_timedelta(1, 's')

        # create non-overlapping time blocks of requested size from the timerange start to end
        blocks = pd.date_range(start, end, freq=f'{time_block}S')
        blocks = blocks.to_frame().reset_index(drop=True).rename(columns={0: 'start'})
        blocks['end'] = blocks.start + pd.to_timedelta(time_block, unit='s')
        blocks['pid'] = p
        blocks = blocks.reset_index(names=['block_id'])

        # map each data point to a time block group
        acc['block_group'] = acc.swifter.apply(
            lambda x: blocks[(x.sec_group >= blocks.start) & (x.sec_group < blocks.end)].block_id.item(), axis=1)

        # we then drop any time block with less than time block size * freq samples
        to_drop = acc.block_group.value_counts()[acc.block_group.value_counts() < (time_block * sample_freq)].index
        acc = acc[~acc.block_group.isin(to_drop)]
        blocks = blocks[blocks.block_id.isin(acc.block_group.unique())]

        # get the alcohol level for each time block based on block end time
        blocks['TAC'] = df.loc[
            blocks.swifter.apply(lambda x: np.abs(x.end - df.timestamp).argmin(), axis=1), 'TAC_Reading'].values

        # save blocks (this has our response variable and maps to data)
        blocks.to_csv(f'./data/{p}_y.csv')

        # we'll partition by block group to create our data points, a 3 x time_block*sample_freq matrix with the accel
        # axes, x, y, & z, in first dim
        if not os.path.exists(f'./data/accel_data'):
            os.makedirs(f'./data/accel_data')

        for bg in acc.block_group.unique():
            ex = acc[acc.block_group == bg].sort_values(by=['time'])[['time', 'x', 'y', 'z']].set_index(
                'time').T.values.astype('float32')
            np.save(f'./data/accel_data/{p}_{bg}', ex)

    # get all y data TODO: save this frame, delete others!
    y = combine_y(pids)
    y.to_csv('./data/y_data_full.csv')

    # split the data into train & test
    train_ids, test_ids = train_test_split(y.index, test_size=0.25, random_state=4, stratify=y.pid)

    # move the files to their correct location
    if not os.path.exists('./data/train'):
        os.makedirs('./data/train')
    if not os.path.exists('./data/test'):
        os.makedirs('./data/test')

    for fpath in tqdm((pd.Series(train_ids) + '.npy'), desc="moving files"):
        source = './data/accel_data/' + fpath
        dest = './data/train/' + fpath
        os.replace(source, dest)

    for fpath in tqdm((pd.Series(test_ids) + '.npy'), desc="moving test files"):
        source = './data/accel_data/' + fpath
        dest = './data/test/' + fpath
        os.replace(source, dest)

    # make sure files were moved correctly
    assert set(f for f in os.listdir('./data/train') if f[-3:] == 'npy') == set(train_ids + '.npy'), 'Train move error'
    assert set(f for f in os.listdir('./data/test') if f[-3:] == 'npy') == set(test_ids + '.npy'), 'Test move error'
    for p in pids:
        os.remove(f'./data/{p}_y.csv')
    os.rmdir('./data/accel_data')


def combine_y(pids) -> pd.DataFrame:
    """
    Combines all time block data frames and creates binary response variable 'over_limit'.

    :param pids:
        list of all pids
    :return:
        DataFrame of all response data indexed by pid_blockid
    """
    frames = (pd.read_csv(f'./data/{ii}_y.csv', index_col=0, parse_dates=['start', 'end']) for ii in pids)
    full = pd.concat(frames, ignore_index=True)

    full['over_limit'] = (full.TAC >= 0.08).values.astype(float)
    full.index = full.pid + '_' + full.block_id.astype(str)

    return full


if __name__ == "__main__":
    main()
