#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import argparse
import logging
import os
import sys


def hms_to_seconds(hms):
    hour = hms.split(':')[0]
    minute = hms.split(':')[1]
    second = hms.split(':')[2].split('.')[0]

    # .xx (10 ms order)
    ms10 = hms.split(':')[2].split('.')[1]

    # total seconds
    seconds = int(hour) * 3600 + int(minute) * 60 + int(second)

    return '{:07d}'.format(int(str(seconds) + ms10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', help='JSON transcription file')
    args = parser.parse_args()

    # logging info
    log_format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    logging.debug("reading %s", args.json)
    with open(args.json, 'rt', encoding="utf-8") as f:
        j = json.load(f)

    session_id = os.path.basename(args.json).split('.json')[0]
    for x in j:
        speaker_id = x['speakerID']

        start_time = x['startTime']
        end_time = x['endTime']
    
        # remove meta chars and convert to lower
        words = x['words'].replace('"', '')\
                            .replace('.', '')\
                            .replace('?', '')\
                            .replace(',', '')\
                            .replace(':', '')\
                            .replace(';', '')\
                            .replace('!', '').lower()

        # remove multiple spaces
        words = " ".join(words.split())
        
        # Time is already in seconds
        start_time = '{:07d}'.format(int(start_time * 100))
        end_time = '{:07d}'.format(int(end_time * 100))

        uttid = speaker_id + '-' + session_id.upper() + '-' + start_time + '-' + end_time

        sys.stdout.buffer.write((uttid + ' ' + words + '\n').encode("utf-8"))
