#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from __future__ import print_function

import argparse
import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--normalize', type=bool, default=False)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) if not file.startswith('.csv')]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))

    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    train_data = pd.concat(raw_data)
    
    # 学習およびラベル項目を取り出してnd arrayに変換する
    # 第1カラムがラベル、それ以降が説明変数としている
    train_x = train_data.values[:,1:]
    train_y = train_data.values[:,0]

    # モデルのトレーニングを実行
    lr_model = LinearRegression(normalize=args.normalize)
    lr_model.fit(train_x, train_y)    
    
    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(lr_model, os.path.join(args.model_dir, "model.joblib"))
    

def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    print('tuki code1:', model_dir)
    
    lr_model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return lr_model
