from typing import Tuple
import pandas as pd
from psycopg2.extras import execute_batch


class SalaryDetector:
    """
    A class to detect and process salary-related transactions data.
    """

    def __init__(self, conn):
        """Initialize with database connection"""
        self.conn = conn

    def get_credit_transactions(self) -> pd.DataFrame:
        """
        Fetch credit transactions from the database with amount >= 70000.
        Returns a pandas DataFrame.
        """
        query = """
            SELECT ob_transaction_id,
                customer_id,
                merchant_name,
                amount,
                ob_transaction_type,
                ob_transaction_description,
                ob_transaction_timestamp
            FROM mv_transactions
            WHERE ob_transaction_type = 'CREDIT'
            AND amount >= 70000
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                # Extract column names from cursor.description
                colnames = [desc[0] for desc in cur.description]

            return pd.DataFrame(rows, columns=colnames)

        except Exception as e:
            print(f"An error occurred: {e}")
            return pd.DataFrame()


    def detect_monthly_salary(
        self,
        df: pd.DataFrame,
        customer_id: int,
        min_amount: float = 70000.0,
        day_interval_range: Tuple[int, int] = (25, 35),
        amount_tolerance: float = 0.1,
        min_occurrences: int = 3,
    ) -> pd.DataFrame:
        """
        Detect potential monthly salary transactions for a single customer.

        Steps:
        1. Filter transactions (CREDIT, minimum amount, etc.).
        2. Group by merchant (same merchant implies same potential employer).
        3. Within each merchant, sort by date and:
        - Compute day intervals between consecutive transactions.
        - Check if intervals ~ monthly (28–35 days).
        - Check if amounts are similar (within `amount_tolerance` of median or each other).
        4. Return all transactions that fit a recurring pattern (>= `min_occurrences`).
        """

        # --- 1) Filter to relevant transactions ---
        customer_df = df[
            (df["customer_id"] == customer_id)
            & (df["ob_transaction_type"] == "CREDIT")
            & (df["amount"] >= min_amount)
        ].copy()

        if customer_df.empty:
            return pd.DataFrame()

        # Convert timestamps to datetime
        if not pd.api.types.is_datetime64_any_dtype(
            customer_df["ob_transaction_timestamp"]
        ):
            customer_df["ob_transaction_timestamp"] = pd.to_datetime(
                customer_df["ob_transaction_timestamp"]
            )

        # Sort globally (this also helps the grouping step)
        customer_df.sort_values("ob_transaction_timestamp", inplace=True)

        # --- 2) Group by merchant ---
        results = []

        grouped = customer_df.groupby("merchant_name", dropna=False)

        for merchant, grp in grouped:
            # Sort by date within this merchant group
            grp = grp.sort_values("ob_transaction_timestamp").copy()
            if len(grp) < min_occurrences:
                continue

            # --- 3) Compute day intervals between consecutive transactions ---
            grp["days_since_last"] = grp["ob_transaction_timestamp"].diff().dt.days

            # A quick approach: we check if a large majority of intervals are within 28–35 days
            valid_intervals = (grp["days_since_last"] >= day_interval_range[0]) & (
                grp["days_since_last"] <= day_interval_range[1]
            )

            # If at least (min_occurrences - 1) intervals are valid, there's a good chance of monthly pattern.
            # Because for N transactions, there are N-1 intervals.
            if valid_intervals.sum() < (min_occurrences - 1):
                continue

            # --- 4) Check if amounts are similar ---
            median_amount = grp["amount"].median()

            # Tolerance check: amount must be within e.g. 10% if amount_tolerance=0.1
            lower_bound = median_amount * (1 - amount_tolerance)
            upper_bound = median_amount * (1 + amount_tolerance)

            grp["amount_in_range"] = grp["amount"].between(lower_bound, upper_bound)

            # We'll consider this a valid merchant if at least min_occurrences transactions
            # are within the range and the monthly interval condition.
            # We'll combine both conditions: day interval pattern + amount similarity.
            # For day interval pattern, we need consecutive intervals in range. Let's
            # build a small rolling approach to see if we can get a run of `min_occurrences` valid intervals.

            # We'll look at sequences:
            #   - We want at least min_occurrences transactions in a row that have 'days_since_last' in valid range
            #   - And amounts are in tolerance range
            # More simply, let's do a pass over the transactions to find consecutive monthly intervals.

            candidate_indices = grp.index.tolist()
            valid_streak_indices = []

            # Start from the first row (except it doesn't have a days_since_last), then move forward
            # We want a sliding window to see if each consecutive pair is valid. We'll track runs.
            streak = [candidate_indices[0]]
            for i in range(1, len(grp)):
                current_idx = candidate_indices[i]
                prev_idx = candidate_indices[i - 1]

                # Check if current transaction days_since_last is in monthly range:
                in_interval = grp.loc[current_idx, "days_since_last"]
                if (
                    in_interval >= day_interval_range[0]
                    and in_interval <= day_interval_range[1]
                ):
                    # Now also check if both this and the previous transaction's amounts are in range
                    if (
                        grp.loc[current_idx, "amount_in_range"]
                        and grp.loc[prev_idx, "amount_in_range"]
                    ):
                        streak.append(current_idx)
                    else:
                        # Streak breaks
                        if len(streak) >= min_occurrences:
                            valid_streak_indices.extend(streak)
                        streak = [current_idx]
                else:
                    # Streak breaks
                    if len(streak) >= min_occurrences:
                        valid_streak_indices.extend(streak)
                    streak = [current_idx]

            # End of loop, check the final streak
            if len(streak) >= min_occurrences:
                valid_streak_indices.extend(streak)

            valid_streak_indices = list(set(valid_streak_indices))  # unique
            if not valid_streak_indices:
                continue

            # We'll add the valid transactions to our result set:
            final_grp = grp.loc[valid_streak_indices].copy()
            results.append(final_grp)

        if not results:
            return pd.DataFrame()

        # Combine all merchant results
        result_df = pd.concat(results).sort_values("ob_transaction_timestamp")

        # drop duplicates if a transaction meets multiple patterns
        result_df.drop_duplicates(
            subset=["ob_transaction_id"], inplace=True, ignore_index=True
        )

        # Pick the columns
        columns_to_return = [
            "customer_id",
            "ob_transaction_id",
            "merchant_name",
            "amount",
            "ob_transaction_description",
            "ob_transaction_timestamp",
        ]
        for col in columns_to_return:
            if col not in result_df.columns:
                result_df[col] = None

        return result_df[columns_to_return]

    def save_salary_transactions(self, df: pd.DataFrame) -> None:
        """
        Save detected salary transactions to customer_salary table.
        Creates table if it doesn't exist.
        """
        if df.empty:
            return

        create_table_query = """
        CREATE TABLE IF NOT EXISTS customer_salary (
            customer_id INTEGER,
            ob_transaction_id VARCHAR(255) PRIMARY KEY,
            merchant_name VARCHAR(255),
            amount DECIMAL(15,2),
            ob_transaction_description TEXT,
            ob_transaction_timestamp TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        upsert_query = """
        INSERT INTO customer_salary (
            customer_id,
            ob_transaction_id,
            merchant_name,
            amount,
            ob_transaction_description,
            ob_transaction_timestamp
        ) VALUES (
            %(customer_id)s,
            %(ob_transaction_id)s,
            %(merchant_name)s,
            %(amount)s,
            %(ob_transaction_description)s,
            %(ob_transaction_timestamp)s
        )
        ON CONFLICT (ob_transaction_id)
        DO UPDATE SET
            merchant_name = EXCLUDED.merchant_name,
            amount = EXCLUDED.amount,
            ob_transaction_description = EXCLUDED.ob_transaction_description,
            ob_transaction_timestamp = EXCLUDED.ob_transaction_timestamp
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(create_table_query)
                records = df.to_dict("records")
                execute_batch(cur, upsert_query, records)
        except Exception as e:
            print(f"Error saving salary transactions: {e}")

    def process_customer_salary(self) -> None:
        """
        Process complete salary detection pipeline for a customer.
        Detects and saves salary transactions.
        """
        credit_df = self.get_credit_transactions()
        customer_ids = credit_df["customer_id"].unique()
        all_salary_transactions = []

        # Process each customer and append salary transactions to the list
        for customer_id in customer_ids:
            salary_transactions = self.detect_monthly_salary(credit_df, customer_id)
            if not salary_transactions.empty:
                all_salary_transactions.extend(salary_transactions.to_dict("records"))
                print(
                    f"Salary transactions for customer {customer_id} appended to list"
                )
            else:
                print(f"No salary transactions detected for customer {customer_id}")

        # Create a DataFrame from the list of all salary transactions
        all_salary_transactions_df = pd.DataFrame(all_salary_transactions)
        self.save_salary_transactions(all_salary_transactions_df)
