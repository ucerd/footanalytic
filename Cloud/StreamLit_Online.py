import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import os
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


# Set up the title and file upload section
st.title("Welcome to Foot Weight Analytics Using Machine Learning")
st.subheader("Foot Scan")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your Dataset as CSV file(Left Foot)", type="csv")

# Load the dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.to_csv('dataset.csv', index=False)
    st.success("Dataset uploaded successfully!")
else:
    if os.path.exists('dataset.csv'):
        df = pd.read_csv('dataset.csv')
    else:
        st.warning("Please upload a dataset to proceed.")
        df = None

# Function to categorize foot regions (model train)
def categorize_foot_region(df):
    forefoot = df.iloc[0:5, :].mean().mean()
    medial_foot = df.iloc[5:10, :].mean().mean()
    rear_foot = df.iloc[10:16, :].mean().mean()

    forefoot_healthy = 28 <= forefoot <= 35
    medial_foot_healthy = 14 <= medial_foot <= 19
    rear_foot_healthy = 46 <= rear_foot <= 54

    if forefoot_healthy and medial_foot_healthy and rear_foot_healthy:
        return "Structural issues May Occurs"
    else:
        return "Healthy"
        
# Display the dataframe if available
if df is not None:
    st.dataframe(df)

    # Categorize and display foot health
    foot_health = categorize_foot_region(df)
    st.subheader(f"Foot Health Status: {foot_health}")

    # 2D Visualization
    st.subheader("2D Left Foot Shape Visualization")
    data = np.array(df)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.contourf(data, cmap='viridis', levels=np.linspace(0, 100, 11), origin='lower')
    ax.set_xlabel('Foot Regions')
    ax.set_ylabel('Data Rows')
    ax.set_title('2D Left Foot Shape Visualization', fontsize=16)
    x_ticks = np.arange(len(data[0]))
    x_tick_labels = ['C' + str(i + 1) for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
    y_ticks = np.arange(data.shape[0])
    y_tick_labels = ['Row ' + str(i + 1) for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Weight Distribution', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

    # 3D Visualization
    st.subheader("3D Left Foot Shape Visualization")
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    z = data
    trace = go.Surface(x=x, y=y, z=z, colorscale='Inferno')
    layout = go.Layout(
        title='3D Left Foot Shape Visualization',
        scene=dict(
            xaxis=dict(title='Foot Regions', tickvals=np.arange(data.shape[1]), ticktext=['C' + str(i + 1) for i in range(data.shape[1])]),
            yaxis=dict(title='Data Rows', tickvals=np.arange(data.shape[0]), ticktext=['Row ' + str(i + 1) for i in range(data.shape[0])]),
            zaxis=dict(title='Weight Distribution'),
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600
    )
    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig)
##############################################################

# Upload CSV file
uploaded_file = st.file_uploader("Upload your Dataset as CSV file(Right Foot)", type="csv")

# Load the dataset
if uploaded_file is not None:
    df1 = pd.read_csv(uploaded_file)
    df1.to_csv('dataset.csv', index=False)
    st.success("Dataset uploaded successfully!")
else:
    if os.path.exists('dataset.csv'):
        df1 = pd.read_csv('dataset.csv')
    else:
        st.warning("Please upload a dataset to proceed.")
        df1 = None

# Function to categorize foot regions
def categorize_foot_region(df1):
    forefoot = df1.iloc[0:5, :].mean().mean()
    medial_foot = df1.iloc[5:10, :].mean().mean()
    rear_foot = df1.iloc[10:15, :].mean().mean()

    forefoot_healthy = 14 <= forefoot <= 19
    medial_foot_healthy = 28 <= medial_foot <= 38
    rear_foot_healthy = 46 <= rear_foot <= 54

    if forefoot_healthy and medial_foot_healthy and rear_foot_healthy:
        return "Healthy"
    else:
        return "Structural issues May Occurs"

# Display the dataframe if available
if df1 is not None:
    st.dataframe(df1)

    # Categorize and display foot health
    foot_health = categorize_foot_region(df1)
    st.subheader(f"Foot Health Status: {foot_health}")

    # 2D Visualization
    st.subheader("2D Right Foot Shape Visualization")
    data = np.array(df1)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.contourf(data, cmap='viridis', levels=np.linspace(0, 100, 11), origin='lower')
    ax.set_xlabel('Foot Regions')
    ax.set_ylabel('Data Rows')
    ax.set_title('2D Right Foot Shape Visualization', fontsize=16)
    x_ticks = np.arange(len(data[0]))
    x_tick_labels = ['C' + str(i + 1) for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
    y_ticks = np.arange(data.shape[0])
    y_tick_labels = ['Row ' + str(i + 1) for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Weight Distribution', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

    # 3D Visualization
    st.subheader("3D Right Foot Shape Visualization")
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    z = data
    trace = go.Surface(x=x, y=y, z=z, colorscale='Inferno')
    layout = go.Layout(
        title='3D Right Foot Shape Visualization',
        scene=dict(
            xaxis=dict(title='Foot Regions', tickvals=np.arange(data.shape[1]), ticktext=['C' + str(i + 1) for i in range(data.shape[1])]),
            yaxis=dict(title='Data Rows', tickvals=np.arange(data.shape[0]), ticktext=['Row ' + str(i + 1) for i in range(data.shape[0])]),
            zaxis=dict(title='Weight Distribution'),
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600
    )
    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig)

##############################################################


# Show the plot
plt.show()
with st.sidebar:
    st.image("http://www.footanalytic.com/images/foot_print_logo.png")
    st.title("Foot Weight Analytics")
    st.info("This project powered by Ucerd Upload your data and choose the desired operation: EDA, Data Preparation, or Modeling")
    choice = st.radio("Choose the Desired operation", ["Upload your csv(Foot Scan)", "Perform EDA(Foot Analysis)", "Data Preparing(Grafiting Your Data)", "Perform modeling"])



if choice == "Perform EDA(Foot Analysis)":
    st.title("Exploratory Data Analysis")

    eda_choise = st.selectbox('Pick the operation you want',['','Show shape','Show data type','Show messing values','Summary',
                                                             'Show columns','Show selected columns','Show Value Counts'])
    if eda_choise =='Show shape' :
        st.write(df.shape)

    if eda_choise =='Show data type' :
        st.write(df.dtypes)

    if eda_choise == 'Show messing values' :
        st.write(df.isna().sum())

    if eda_choise =='Summary' :
        st.write(df.describe())

    if eda_choise =='Show columns' :
        all_columns = df.columns
        st.write(all_columns)

    if eda_choise =='Show selected columns' :
        selected_columns = st.multiselect('Select desired columns',df.columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    if eda_choise =='Show Value Counts' :

        try:

            selected_columns = st.multiselect('Select desired columns', df.columns)
            new_df = df[selected_columns]
            st.write(new_df.value_counts().rename(index='Value'))

        except:
            pass


    plot_choice = st.selectbox('Select type of plot you want :',['','Box Plot','Correlation Plot','Pie Plot',
                                                                 'Scatter Plot','Bar Plot'])


    if plot_choice == 'Box Plot' :
        column_to_plot = st.selectbox("Select 1 Column", df.columns)
        fig = px.box(df,y=column_to_plot)
        st.plotly_chart(fig)

    if plot_choice =='Correlation Plot' :
        plt.figure(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        st.pyplot(plt)

    if plot_choice =='Pie Plot' :
        column_to_plot = st.selectbox("Select 1 Column", df.columns)
        value_counts = df[column_to_plot].value_counts()
        fig, ax = plt.subplots()
        ax.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis('equal')
        st.write(fig)

    if plot_choice =='Scatter Plot' :

        try :

            selected_columns = st.multiselect('Select two columns',df.columns)
            first_column = selected_columns[0]
            second_column = selected_columns[1]
            fig = px.scatter(df, x=first_column, y=second_column)
            fig.update_layout(title="Scatter Plot", xaxis_title=first_column, yaxis_title=second_column)
            st.plotly_chart(fig)

        except:
            pass

    if plot_choice == 'Bar Plot':

        try :

            selected_columns = st.multiselect('Select columns', df.columns)
            first_column = selected_columns[0]
            second_column = selected_columns[1]

            fig = px.bar(df, x=first_column, y=second_column, title='Bar Plot')
            st.plotly_chart(fig)

        except :
            pass



if choice == "Data Preparing(Grafiting Your Data)" :

    st.title('Preparing the data before machine learning modeling')


    want_to_drop = st.selectbox('Do you want to drop any columns ?',['','Yes','No'])

    if want_to_drop == 'No':

        st.warning('It is recommended to drop columns such as name, customer ID, etc.')

    if want_to_drop == 'Yes':

        columns_to_drop = st.multiselect('Select columns to drop', df.columns)
        if columns_to_drop  :
            df = df.drop(columns_to_drop, axis=1)
            st.success('Columns dropped successfully.')
            st.dataframe(df)


    encoder_option = st.selectbox('Do you want to encode your data ?',['','Yes','No'])

    if encoder_option == 'No' :

        st.write('OK, Please processed to next step')

    if encoder_option == 'Yes' :

        encoder_columns = st.multiselect('Please pick the columns you want to encode',df.columns)
        encoder_type = st.selectbox('Please pick the type of encoder you want to use', ['','Label Encoder','One Hot Encoder'])

        if encoder_type == 'Label Encoder' :

            encoder = LabelEncoder()
            df[encoder_columns] = df[encoder_columns].apply(encoder.fit_transform)
            st.success('Columns encoded successfully.')
            st.dataframe(df)

        if encoder_type == 'One Hot Encoder':

            df = pd.get_dummies(df, columns=encoder_columns, prefix=encoder_columns,drop_first=True)
            st.success('Columns encoded successfully.')
            st.dataframe(df)


    fill_option = st.selectbox('Is there any missing data you want to fill ?', ['', 'Yes', 'No'])

    if fill_option == 'No':

        st.write('OK, Please processed to next step')

    if fill_option == 'Yes':

        encoder_columns = st.multiselect('Please pick the columns you want to fill', df.columns)
        encoder_type = st.selectbox('Please pick the type of filling you want to use', ['','Mean','Median','Most frequent'])

        try:

            if encoder_type == 'Mean' :

                imputer = SimpleImputer(strategy='mean')
                df[encoder_columns] = np.round(imputer.fit_transform(df[encoder_columns]),1)
                st.success('Selected columns filled successfully')
                st.dataframe(df)

            if encoder_type == 'Median' :

                imputer = SimpleImputer(strategy='median')
                df[encoder_columns] = np.round(imputer.fit_transform(df[encoder_columns]),1)
                st.success('Selected columns filled successfully')
                st.dataframe(df)

            if encoder_type == 'Most frequent' :

                imputer = SimpleImputer(strategy='most_frequent')
                df[encoder_columns] = np.round(imputer.fit_transform(df[encoder_columns]),1)
                st.success('Selected columns filled successfully')
                st.dataframe(df)

        except :
            pass

    scaling_option = st.selectbox('Do you want to scale your data ?',['','Yes','No'])

    if scaling_option == 'No' :

        st.write('OK, Please processed to next step')
        st.dataframe(df)

    if scaling_option == 'Yes' :

        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
        st.success('The Data Frame has been successfully scaled')
        st.dataframe(df_scaled)

        try :

            df = df_scaled.to_csv('dataset.csv', index=None)
        except :
            pass

if choice == "Perform modeling":

    st.title('It is time for Machine Learning modeling')
    df = pd.read_csv('dataset.csv', index_col=None)

    target_choices = [''] + df.columns.tolist()

    try :
        target = st.selectbox('Choose your target variable', target_choices)
        X = df.drop(columns=target)
        y = df[target]
        st.write('Your Features are', X)
        st.write('Your Target is', y)

        test_size = st.select_slider('Pick the test size you want', range(1, 100, 1))
        st.warning('It is recommended to pick a number between 10 and 30 ')
        test_size_fraction = test_size / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_fraction, random_state=42)
        st.write('Shape of training data is :', X_train.shape)
        st.write('Shape of testing data is :', X_test.shape)

    except :
        pass

    task_type = st.selectbox('Choose type of task you want to apply', ['','Classification', 'Regression'])
    modeling_choice = st.selectbox('Do you want Auto modeling or you want to choose the model ?',
                                   ['','Auto modeling','Manual modeling'])

    if task_type == 'Classification':

        if modeling_choice == 'Auto modeling':

            from pycaret.classification import *

            if st.button('Run Modelling'):

                setup(df, target=target, verbose=False)
                setup_df = pull()
                st.info("This is the ML experiment settings")
                st.dataframe(setup_df)
                best_model = compare_models()
                compare_df = pull()
                st.info("This is your ML model")
                st.dataframe(compare_df)
                save_model(best_model, 'best_model')

                with open('best_model.pkl', 'rb') as model_file:
                    st.download_button('Download the model', model_file, 'best_model.pkl')

        try :
            if modeling_choice == 'Manual modeling' :

                algo_type = st.selectbox('Please choose which type of algorithm you want to use',
                                         ['','Logistic Regression','Decision Trees','Random Forest','SVC','KNN'])

                if algo_type == 'Logistic Regression' :

                    from sklearn.linear_model import LogisticRegression

                    clf = LogisticRegression(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)


                if algo_type == 'Decision Trees' :

                    from sklearn.tree import DecisionTreeClassifier

                    clf = DecisionTreeClassifier(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                if algo_type == 'Random Forest' :

                    from sklearn.ensemble import RandomForestClassifier

                    clf = RandomForestClassifier(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                if algo_type == 'SVC' :

                    from sklearn.svm import SVC

                    clf = SVC(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                if algo_type == 'KNN' :

                    from sklearn.neighbors import KNeighborsClassifier

                    clf = KNeighborsClassifier()
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

        except :
            st.warning('Please choose a valid binary or multivalued target for your classification problem')

            evaluation_type = st.selectbox('Choose type of evaluation metrics ',['','Accuracy','Confusion Matrix',
                                                                                 'Precision, Recall, and F1-score'])

            if evaluation_type == 'Accuracy' :

                from sklearn.metrics import accuracy_score

                accuracy = accuracy_score(y_test, y_pred)
                st.write("Accuracy:", accuracy)

            if evaluation_type == 'Confusion Matrix' :

                from sklearn.metrics import confusion_matrix

                cm = confusion_matrix(y_test, y_pred)
                st.write("Confusion Matrix:")
                st.dataframe(cm)

            if evaluation_type == 'Precision, Recall, and F1-score' :

                from sklearn.metrics import precision_score, recall_score, f1_score

                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                metrics_dict = {
                    "Metric": ["Precision", "Recall", "F1-Score"],
                    "Value": [precision, recall, f1]
                }
                metrics_df = pd.DataFrame(metrics_dict)
                st.dataframe(metrics_df)


            try :

                model_filename = "clf.pkl"
                with open(model_filename, "wb") as model_file:
                    pickle.dump(clf, model_file)

                st.download_button('Download the model', open(model_filename, 'rb').read(), 'clf.pkl')

            except :
                pass

    if task_type == 'Regression':

        if modeling_choice == 'Auto modeling':

            from pycaret.regression import *

            if st.button('Run Modelling'):

                setup(df, target=target, verbose=False)
                setup_df = pull()
                st.info("This is the ML experiment settings")
                st.dataframe(setup_df)
                best_model = compare_models()
                compare_df = pull()
                st.info("This is your ML model")
                st.dataframe(compare_df)
                save_model(best_model, 'best_model')

                with open('best_model.pkl', 'rb') as model_file:
                    st.download_button('Download the model', model_file, 'best_model.pkl')


        if modeling_choice == 'Manual modeling' :

            algo_type = st.selectbox('Please choose which type of algorithm you want to use',
                                     ['','Linear Regression','Ridge','SVR','Random Forest'])

            if algo_type == 'Linear Regression' :

                from sklearn.linear_model import LinearRegression

                rg = LinearRegression()
                rg.fit(X_train, y_train)
                y_pred = rg.predict(X_test)


            if algo_type == 'Ridge' :

                from sklearn.linear_model import Ridge

                rg = Ridge()
                rg.fit(X_train, y_train)
                y_pred = rg.predict(X_test)


            if algo_type == 'SVR' :

                from sklearn.svm import SVR

                rg = SVR()
                rg.fit(X_train, y_train)
                y_pred = rg.predict(X_test)

            if algo_type == 'Random Forest' :

                from sklearn.ensemble import RandomForestRegressor

                rg = RandomForestRegressor()
                rg.fit(X_train, y_train)
                y_pred = rg.predict(X_test)

            evaluation_type = st.selectbox('Choose type of evaluation metrics ',['','MAE','MSE','r2 score'])

            if evaluation_type == 'MAE' :

                from sklearn.metrics import mean_absolute_error

                MAE = mean_absolute_error(y_test, y_pred)
                st.write("Mean absolute error:", MAE)

            if evaluation_type == 'MSE' :

                from sklearn.metrics import mean_squared_error

                MSE = mean_squared_error(y_test, y_pred)
                st.write("Mean squared error:", MSE)

            if evaluation_type == 'r2 score' :

                from sklearn.metrics import r2_score

                r2 = r2_score(y_test, y_pred)
                st.write("r2 score:", r2)


            try :

                model_filename = "rg.pkl"
                with open(model_filename, "wb") as model_file:
                    pickle.dump(rg, model_file)

                st.download_button('Download the model', open(model_filename, 'rb').read(), 'rg.pkl')

            except :
                pass
