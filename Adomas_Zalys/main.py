import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from datetime import datetime, timedelta
import os
import re
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


DATA_DIR = "data"
TRANS_FILE = os.path.join(DATA_DIR, "transactions.csv")
PORT_FILE = os.path.join(DATA_DIR, "portfolio.csv")
SAVINGS_FILE = os.path.join(DATA_DIR, "savings.csv")
PORT_HISTORY_FILE = os.path.join(DATA_DIR, "portfolio_history.csv")

os.makedirs(DATA_DIR, exist_ok=True)


if not os.path.exists(TRANS_FILE):
    df = pd.DataFrame(columns=["Date", "Description", "Amount", "Category", "Type"])
    df.to_csv(TRANS_FILE, index=False)

if not os.path.exists(PORT_FILE):
    pf = pd.DataFrame(columns=["Ticker", "Shares", "AmountInvested", "CurrentValue"])
    pf.to_csv(PORT_FILE, index=False)

if not os.path.exists(SAVINGS_FILE):
    sv = pd.DataFrame(columns=["Date", "Change", "Reason", "Type"])
    sv.to_csv(SAVINGS_FILE, index=False)

if not os.path.exists(PORT_HISTORY_FILE):
    ph = pd.DataFrame(columns=["Date", "Shares", "Price", "Value"])
    ph.to_csv(PORT_HISTORY_FILE, index=False)


CATEGORY_RULES = {
    "Food": ["food", "coffee", "restaurant", "grocery", "supermarket", "lunch", "dinner", "pizza"],
    "Transport": ["transport", "bus", "taxi", "gas", "train"],
    "Shopping": ["amazon", "clothes", "store", "shopping"],
    "Bills": ["electricity", "water", "internet", "rent", "bill"],
    "Entertainment": ["movie", "game", "netflix", "spotify"],
    "Salary": ["salary"],
    "Freelance": ["freelance"],
    "Bonus": ["bonus"],
    "Gift": ["gift"],

 
    "Income": ["income"],
}


def parse_transaction(text):
    text_lower = text.lower().strip()


    nums = re.findall(r"\d+\.?\d*", text)
    if not nums:
        print("No numeric amount found. Try again.")
        return None
    amount = float(nums[0])


    income_keywords = ["+", "salary", "freelance", "bonus", "gift", "income", "earned"]
    ttype = "Income" if any(k in text_lower for k in income_keywords) else "Expense"

    if ttype == "Expense":
        amount = -abs(amount)


    category = "Other"
    for cat, keywords in CATEGORY_RULES.items():
        if any(word in text_lower for word in keywords):
            category = cat
            break


    if category == "Other" and ttype == "Income":
        category = "Income"

    return {
        "Date": datetime.today().strftime("%Y-%m-%d"),
        "Description": text,
        "Amount": amount,
        "Category": category,
        "Type": ttype
    }

def add_transaction(transaction):
    df = pd.read_csv(TRANS_FILE)
    df = pd.concat([df, pd.DataFrame([transaction])], ignore_index=True)
    df.to_csv(TRANS_FILE, index=False)


def modify_savings(amount, reason="Manual"):
    sv = pd.read_csv(SAVINGS_FILE)
    sv = pd.concat([sv, pd.DataFrame([{
        "Date": datetime.today().strftime("%Y-%m-%d"),
        "Change": amount,
        "Reason": reason,
        "Type": "Manual"
    }])], ignore_index=True)
    sv.to_csv(SAVINGS_FILE, index=False)


def update_monthly_savings():
    df = pd.read_csv(TRANS_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    today = datetime.today()
    month_mask = df['Date'].dt.month == today.month
    month_df = df.loc[month_mask]
    if month_df.empty:
        print("No transactions this month for savings update.")
        return
    income = month_df[month_df['Type'] == "Income"]['Amount'].sum()
    expense = month_df[month_df['Type'] == "Expense"]['Amount'].sum()
    leftover = income + expense
    if leftover != 0:
 
        add_transaction({"Date": datetime.today().strftime("%Y-%m-%d"),
                         "Description": "Monthly leftover transfer to savings",
                         "Amount": -leftover,
                         "Category": "Savings",
                         "Type": "Expense"})
        modify_savings(leftover, reason="Monthly leftover")
    print(f"Savings updated by monthly leftover: {leftover:.2f}")


def transfer_savings(amount):
    df = pd.read_csv(TRANS_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    total_income = df[df["Type"] == "Income"]["Amount"].sum()
    total_expense = df[df["Type"] == "Expense"]["Amount"].sum()
    net_spending = total_income + total_expense
    sv = pd.read_csv(SAVINGS_FILE)
    savings_total = sv['Change'].sum() if not sv.empty else 0

    if amount > 0:
        if net_spending >= amount:
            add_transaction({"Date": datetime.today().strftime("%Y-%m-%d"),
                             "Description": "Transfer to savings",
                             "Amount": -amount,
                             "Category": "Savings",
                             "Type": "Expense"})
            modify_savings(amount, reason="Transfer from balance")
            print(f"Moved {amount:.2f} to savings from balance.")
        else:
            print("Not enough money in spending account to move to savings.")
    else:
        if savings_total >= abs(amount):
            add_transaction({"Date": datetime.today().strftime("%Y-%m-%d"),
                             "Description": "Transfer from savings",
                             "Amount": abs(amount),
                             "Category": "Savings",
                             "Type": "Income"})
            modify_savings(amount, reason="Transfer to balance")
            print(f"Moved {abs(amount):.2f} from savings to balance.")
        else:
            print("Not enough savings to transfer.")


def show_savings():
    sv = pd.read_csv(SAVINGS_FILE)
    total = sv['Change'].sum() if not sv.empty else 0
    print(f"Savings: {total:.2f}")


def get_sp500_price():
    sp500 = yf.Ticker("^GSPC")
    hist = sp500.history(period="1d")
    if hist.empty:
        return 0
    return hist['Close'].iloc[-1]


def get_portfolio_value():
    pf = pd.read_csv(PORT_FILE)
    if pf.empty:
        return 0, 0, 0
    shares = pf['Shares'].sum()
    price_per_share = get_sp500_price()
    total_value = shares * price_per_share
    return total_value, price_per_share, shares


def record_portfolio_snapshot():
    pf = pd.read_csv(PORT_FILE)
    if pf.empty:
        return
    shares = pf.at[0, 'Shares']
    price = get_sp500_price()
    value = shares * price

    snap = pd.DataFrame([{
        "Date": datetime.today().strftime("%Y-%m-%d"),
        "Shares": shares,
        "Price": price,
        "Value": value
    }])

    hist = pd.read_csv(PORT_HISTORY_FILE)
    hist = pd.concat([hist, snap], ignore_index=True)
    hist.to_csv(PORT_HISTORY_FILE, index=False)


def buy_sp500(shares):
    price = get_sp500_price()
    total_cost = shares * price
    add_transaction({"Date": datetime.today().strftime("%Y-%m-%d"),
                     "Description": "Buy SP500",
                     "Amount": -total_cost,
                     "Category": "Investment",
                     "Type": "Expense"})
    pf = pd.read_csv(PORT_FILE)
    if pf.empty:
        pf = pd.DataFrame([{"Ticker": "SP500", "Shares": shares,
                            "AmountInvested": total_cost, "CurrentValue": total_cost}])
    else:
        pf.at[0, 'Shares'] += shares
        pf.at[0, 'AmountInvested'] += total_cost
    pf.to_csv(PORT_FILE, index=False)

    record_portfolio_snapshot()

    print(f"Bought {shares} shares at ${price:.2f} each, total ${total_cost:.2f}")


def sell_sp500(shares):
    pf = pd.read_csv(PORT_FILE)
    if pf.empty or pf.at[0, 'Shares'] < shares:
        print("Not enough shares to sell.")
        return
    price = get_sp500_price()
    total_value = shares * price
    add_transaction({"Date": datetime.today().strftime("%Y-%m-%d"),
                     "Description": "Sell SP500",
                     "Amount": total_value,
                     "Category": "Investment",
                     "Type": "Income"})
    pf.at[0, 'Shares'] -= shares
    pf.at[0, 'AmountInvested'] -= pf.at[0, 'AmountInvested'] / pf.at[0, 'Shares'] * shares if pf.at[0, 'Shares'] > 0 else pf.at[0, 'AmountInvested']
    pf.to_csv(PORT_FILE, index=False)

    record_portfolio_snapshot()

    print(f"Sold {shares} shares at ${price:.2f} each, total ${total_value:.2f}")


def show_balance():
    df = pd.read_csv(TRANS_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    total_income = df[df["Type"] == "Income"]["Amount"].sum()
    total_expense = df[df["Type"] == "Expense"]["Amount"].sum()
    net_spending = total_income + total_expense

    sv = pd.read_csv(SAVINGS_FILE)
    savings_total = sv['Change'].sum() if not sv.empty else 0
    port_val, price, shares = get_portfolio_value()

    print("\n--- BALANCE ---")
    print(f"Income: {total_income:.2f}")
    print(f"Expenses: {-total_expense:.2f}")
    print(f"Net spending: {net_spending:.2f}")
    print(f"Savings: {savings_total:.2f}")
    print(f"Portfolio value (SP500 {shares} shares at ${price:.2f} each): {port_val:.2f}")

def sp500_status(period="1d"):
    sp500 = yf.Ticker("^GSPC")
    hist = sp500.history(period=period)
    if hist.empty:
        print("No SP500 data available.")
        return
    start_price = hist['Close'].iloc[0]
    end_price = hist['Close'].iloc[-1]
    change = (end_price - start_price) / start_price * 100
    start_date = hist.index[0].date()
    end_date = hist.index[-1].date()
    latest = get_sp500_price()

    print("\n--- S&P 500 PERFORMANCE ---")
    print(f"Period: {start_date} → {end_date}")
    print(f"Start price: {start_price:.2f}")
    print(f"End price: {end_price:.2f}")
    print(f"Change: {change:.2f}%")
    print(f"Latest value:{latest:.2f}")


def _make_autopct(values):
    total = sum(values)

    def my_autopct(pct):
        amount = pct / 100.0 * total
        return f"{pct:.1f}%\n{amount:.2f}"
    return my_autopct


def plot_transactions(timeframe="all"):
    df = pd.read_csv(TRANS_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    today = datetime.today().date()

    if timeframe == "today":
        df = df[df['Date'].dt.date == today]
    elif timeframe == "week":
        df = df[df['Date'].dt.date >= (today - timedelta(days=7))]
    elif timeframe == "month":
        df = df[df['Date'].dt.month == today.month]
    elif timeframe.endswith("d"):
        try:
            days = int(timeframe[:-1])
            df = df[df['Date'].dt.date >= (today - timedelta(days=days))]
        except:
            pass

    income_df = df[df['Type'] == "Income"]
    expense_df = df[df['Type'] == "Expense"]

    if not income_df.empty:
        income_sums = income_df.groupby("Category")['Amount'].sum()
        values = income_sums.values
        labels = income_sums.index

        plt.figure(figsize=(5, 5))
        plt.pie(values, labels=labels, autopct=_make_autopct(values))
        plt.title("Income Distribution")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()

    if not expense_df.empty:
        expense_sums = -expense_df.groupby("Category")['Amount'].sum()
        values = expense_sums.values
        labels = expense_sums.index

        plt.figure(figsize=(5, 5))
        plt.pie(values, labels=labels, autopct=_make_autopct(values))
        plt.title("Expenses Distribution")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()


def plot_savings(timeframe="all"):
    sv = pd.read_csv(SAVINGS_FILE)
    if sv.empty:
        print("No savings data.")
        return

    sv['Date'] = pd.to_datetime(sv['Date'])
    sv['Day'] = sv['Date'].dt.date
    today = datetime.today().date()

    if timeframe == "today":
        sv = sv[sv['Day'] == today]
    elif timeframe == "week":
        sv = sv[sv['Day'] >= (today - timedelta(days=7))]
    elif timeframe == "month":
        sv = sv[sv['Date'].dt.month == today.month]
    elif timeframe.endswith("d"):
        try:
            days = int(timeframe[:-1])
            sv = sv[sv['Day'] >= (today - timedelta(days=days))]
        except:
            pass

    if sv.empty:
        print("No savings entries in this timeframe.")
        return

    daily = sv.groupby("Day")['Change'].sum().cumsum()

    plt.figure(figsize=(8, 4))
    plt.plot(daily.index, daily.values, marker='o')
    plt.title("Savings Over Time")
    plt.ylabel("Cumulative Savings")
    plt.xticks(rotation=45)
    plt.xlim(daily.index.min(), daily.index.max())
    plt.tight_layout()
    plt.show()


def plot_portfolio(timeframe="all"):
    if not os.path.exists(PORT_HISTORY_FILE):
        print("No portfolio history recorded yet.")
        return

    df = pd.read_csv(PORT_HISTORY_FILE)
    if df.empty:
        print("No portfolio history recorded yet.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.date
    today = datetime.today().date()

    if timeframe == "today":
        df = df[df['Day'] == today]
    elif timeframe == "week":
        df = df[df['Day'] >= (today - timedelta(days=7))]
    elif timeframe == "month":
        df = df[df['Date'].dt.month == today.month]
    elif timeframe.endswith("d"):
        try:
            days = int(timeframe[:-1])
            df = df[df['Day'] >= (today - timedelta(days=days))]
        except:
            pass

    if df.empty:
        print("No portfolio data in this timeframe.")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(df['Date'], df['Value'], marker='o')
    plt.title("Portfolio Value Over Time")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.xlim(df['Date'].min(), df['Date'].max())
    plt.tight_layout()
    plt.show()


def main():
    print("Welcome to your Finance Assistant!\n")
    print("Commands:")
    print("  Transactions: spent 10 coffee / 100 salary")
    print("  Balance: balance")
    print("  SP500: sp500 [period]")
    print("  Buy SP500: buy sp500 X")
    print("  Sell SP500: sell sp500 X")
    print("  Savings: savings +100 / savings -100 / savings update")
    print("  Plot: plot transactions X / plot savings X / plot portfolio X (X=day/week/month/50d etc)")
    print("  Quit: quit\n")

    while True:
        cmd = input("Enter command: ").strip()
        if not cmd:
            continue
        cmd_lower = cmd.lower()

        if cmd_lower == "quit":
            break
        elif cmd_lower == "savings update":
            update_monthly_savings()
        elif cmd_lower.startswith("savings"):
            parts = cmd.split()
            if len(parts) == 2:
                if parts[1].startswith("+") or parts[1].startswith("-"):
                    try:
                        transfer_savings(float(parts[1]))
                    except:
                        print("Invalid savings command.")
                else:
                    show_savings()
            elif len(parts) == 1:
                show_savings()
            else:
                print("Unknown savings command.")

        elif cmd_lower == "balance":
            show_balance()
        elif cmd_lower.startswith("buy sp500"):
            parts = cmd.split()
            if len(parts) == 3:
                try:
                    buy_sp500(float(parts[2]))
                except:
                    print("Invalid buy command.")
            else:
                print("Format: buy sp500 X")
        elif cmd_lower.startswith("sell sp500"):
            parts = cmd.split()
            if len(parts) == 3:
                try:
                    sell_sp500(float(parts[2]))
                except:
                    print("Invalid sell command.")
            else:
                print("Format: sell sp500 X")

        elif cmd_lower.startswith("sp500"):
            parts = cmd.split()
            period = "1d"
            if len(parts) == 2:
                period = parts[1]
            sp500_status(period)

        elif cmd_lower.startswith("plot"):
            parts = cmd.split()
            if len(parts) == 2:
                plot_transactions(parts[1])
            elif len(parts) == 3:
                if parts[1] == "transactions":
                    plot_transactions(parts[2])
                elif parts[1] == "savings":
                    plot_savings(parts[2])
                elif parts[1] == "portfolio":
                    plot_portfolio(parts[2])
                else:
                    print("Unknown plot type.")
            else:
                print("Usage: plot transactions|savings|portfolio [timeframe]")

        else:
            trans = parse_transaction(cmd)
            if trans:
                add_transaction(trans)
                print("Transaction added!")
            else:
                print("Unknown command.")


if __name__ == "__main__":
    main()
