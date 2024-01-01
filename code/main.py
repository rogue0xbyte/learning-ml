from hypotheses import *
from conditioning import *
from baye import *

def main():
    data = pd.read_csv('Forza_Horizon_Cars.csv')  # Replace 'your_dataset.csv' with your actual dataset file path

    data = preprocess(data)

    features = ['stock_A', 'stock_B', 'stock_C', 'stock_D', 'stock_S1', 'stock_S2',
                'drive_AWD', 'drive_FWD', 'drive_RWD', 'speed', 'handling',
                'acceleration', 'launch', 'braking', 'Offroad',
                'Top_Speed', '0-60_Mph', 'g-force', 'Horse_Power', 'Weight_lbs']

    accuracy = NBPredict(data, features)

    hypotheses = generate_hypotheses(data[features])

    instanceMap = MapInstances(hypotheses, data[features])

    store_output_to_excel(hypotheses, instanceMap, data[features])

    visualize_hypotheses(hypotheses, instanceMap)

if __name__ == '__main__':
    main()