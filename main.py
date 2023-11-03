import numpy as np
import pandas as pd
import customtkinter
from tkcalendar import Calendar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("green")

months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
sub_divs = []
ds = {}
loc_model = {}


def train():
    global months, sub_divs, loc_model
    print("Getting Data...")
    data = pd.read_csv('rainfall in india 1901-2015.csv')
    data = data.drop(["ANNUAL", "Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"], axis=1)
    data.dropna(inplace=True)
    sub_divs = pd.unique(data['SUBDIVISION']).tolist()

    print("Processing Data...")
    for d in sub_divs:
        ds[d] = {}
        for m in months:
            temp = data.query(f'SUBDIVISION=="{d}"')
            ds[d][m] = temp[['YEAR', m]]

    print("Loading Models...")
    try:
        with open("models.pkl", "rb") as f:
            loc_model = joblib.load(f)
        print("Models Loaded Successfully.")
        return

    except Exception as e:
        print("Loading Models Failed.\n", e)

    print("Creating New Models...")
    for d in sub_divs:
        loc_model[d] = {}
        for m in months:
            X_train, X_test, y_train, y_test = train_test_split(ds[d][m][["YEAR"]], ds[d][m][m], test_size=0.25)
            loc_model[d][m] = RandomForestRegressor()
            loc_model[d][m].fit(X_train, y_train)
        print(f"Created Model for {d}.")
    print("Created New Models.\n Saving Model Files...")

    try:
        with open("models.pkl", "wb") as f:
            joblib.dump(loc_model, f)
        print("Saved New Model Files.")

    except:
        print("New Model Files Ready to be used but Saving Failed.")


def pred(place, mth, year):
    return loc_model[place][mth].predict(np.array([[int(year)]]))


class App(customtkinter.CTk):
    APP_NAME = "Weather Forecast System with Machine Learning v1.0.0"
    WIDTH = 400
    HEIGHT = 400

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gdt = None
        self.main_win_btn = None
        self.confirm_date = None
        self.start_cal = None
        self.title(App.APP_NAME)
        self.geometry(str(App.WIDTH) + "x" + str(App.HEIGHT))
        self.minsize(App.WIDTH, App.HEIGHT)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.createcommand('tk::mac::Quit', self.on_closing)

        # ============ Universal Frame Declaration ============

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.top_menu_frame = customtkinter.CTkFrame(master=self)
        self.content_frame = customtkinter.CTkFrame(master=self)

        self.frame_setup()

    def frame_setup(self):
        self.top_menu_frame.grid_columnconfigure(0, weight=1)
        self.top_menu_frame.grid_rowconfigure(0, weight=1)
        self.top_menu_frame.grid_rowconfigure(1, weight=1)
        self.top_menu_frame.grid_rowconfigure(2, weight=1)

        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)

        self.top_menu_frame.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")
        self.content_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")

        # ============ Universal Top Menu Declaration ============
        self.main_win_btn = customtkinter.CTkOptionMenu(master=self.top_menu_frame, values=[x.strip() for x in sub_divs])
        self.main_win_btn.set("Select Sub Division")
        self.main_win_btn.grid(row=0, column=0, padx=30, pady=10, sticky='n')

        self.start_cal = Calendar(self.top_menu_frame, selectmode='day')
        self.start_cal.grid(row=1, column=0, padx=30, pady=10, sticky='n')

        self.confirm_date = customtkinter.CTkButton(self.top_menu_frame, text="Get Weather", command=self.get_data)
        self.confirm_date.grid(row=2, column=0, padx=30, pady=10, sticky='n')

        self.gdt = customtkinter.CTkLabel(master=self.content_frame, text="", font=("Arial", 16))
        self.gdt.grid(row=0, column=0, padx=30, pady=10, sticky='nsew')

    def get_data(self):
        pl = self.main_win_btn.get()
        dt = self.start_cal.get_date()
        mth = int(dt.split('/')[0])
        yer = int(dt.split('/')[2])+2000
        prd = pred(pl, months[mth-1], yer)
        print(pl, dt.split('/'), mth, months[mth-1], yer, prd)
        self.gdt.configure(text=f"Rainfall for the month of\n{months[mth-1]}, {yer} in \n{pl}\nis {prd[0]:.2f}mm.")

    def on_closing(self, event=0):
        self.destroy()
        exit()

    def start(self):
        self.mainloop()


if __name__ == '__main__':
    train()
    app = App()
    app.start()
