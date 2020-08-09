import os

from app import Settings
import numpy as np

from app.other.LoggerFactory import get_logger


class Data:

    class Entry:
        def __init__(self, name, path, y_values):
            self.name = name
            self.path = path
            self.y_values = y_values

    @staticmethod
    def get_sources():
        array = [
            # 1. Source Train frames and Y values.
            Data.Entry(
                'Source',
                Settings.RESOURCE + 'frames/',
                np.loadtxt(
                    Settings.RESOURCE + 'source/train.txt',
                    delimiter=" ")
            )
        ]
        # 2. Stop Frames and Zero-Y values
        location = Settings.RESOURCE + 'frames-stop/'
        for d1 in os.listdir(location):
            path = location + '/' + d1
            array.append(
                Data.Entry(
                    'Stop-{}'.format(d1),
                    path,
                    np.zeros(len(os.listdir(path)))
                )
            )

        # 3. Custom CommaAi data
        location = Settings.CUSTOM + 'Chunk_1/'
        for i, d1 in enumerate(os.listdir(location)):
            for d2 in os.listdir(location + d1):
                outer = os.path.join(location, d1, d2)

                y = np.fromfile(os.path.join(
                    outer,  'processed_log/CAN/speed/value'
                ))
                path = os.path.join(outer,  'frames/')
                assert os.path.isdir(path)

                array.append(
                    Data.Entry(
                        'Custom-Chunk1-{}-{}'.format(i, d2),
                        path,
                        Data.summarize_y(y, len(os.listdir(path)))
                    ),
                )

        return array

    @staticmethod
    def summarize_y(y, frames_len):
        # Should Start from 20th element.
        value = [np.mean(a) for a in np.array_split(
            y[20:], frames_len)]
        assert len(value) == frames_len
        return value


if __name__ == "__main__":
    logger = get_logger()

    def print_info():
        logger.debug('Listing Data Sources!')
        for s in Data.get_sources():

            logger.debug('\n\n{}:'.format(s.name))
            logger.debug(
                '-> Path: {}. Y len: {}.'
                    .format(s.path, len(s.y_values))
            )
            logger.debug(
                '---> Y Max: {}. Y Min: {}.'
                .format(max(s.y_values), min(s.y_values))
            )

    print_info()
