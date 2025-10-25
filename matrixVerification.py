import numpy as np
import json
from functools import wraps

def linearity_decorator(func):
    @wraps(func)
    def wrapper(self, matrix, v1, v2, c):
        # Homogeneity: A * (c * v) == c * (A * v)
        lhs_homogeneous = matrix @ (c * v1)
        rhs_homogeneous = c * (matrix @ v1)
        is_homogeneous = np.allclose(lhs_homogeneous, rhs_homogeneous)

        # Additivity: A * (v1 + v2) == A * v1 + A * v2
        lhs_additive = matrix @ (v1 + v2)
        rhs_additive = matrix @ v1 + matrix @ v2
        is_additive = np.allclose(lhs_additive, rhs_additive)

        if not is_homogeneous:
            raise ValueError("Transformation is not homogeneous: A*(c*v) != c*(A*v).")
        if not is_additive:
            raise ValueError("Transformation is not additive: A*(v1+v2) != A*v1 + A*v2.")

        return func(self, matrix, v1, v2, c)
    return wrapper

def homogeneity_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        matrix = args[1] if len(args) > 1 else args[0]
        is_homogeneous = all(len(set(row)) == 1 for row in matrix)
        if not is_homogeneous:
            raise ValueError("Matrix is not homogeneous: all elements in each row must be the same.")
        return func(*args, **kwargs)
    return wrapper

def additive_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        matrix = args[1] if len(args) > 1 else args[0]
        is_homogeneous = all(len(set(row)) == 1 for row in matrix)
        if not is_homogeneous:
            raise ValueError("Matrix is not additive: it must be homogeneous.")
        return func(*args, **kwargs)
    return wrapper

class matrixVerification:
    @linearity_decorator
    def is_linear(self, matrix, v1, v2, c):
        return {"linear": True, "homogeneous": True, "additive": True}

    @homogeneity_decorator
    def homogeneity(self, matrix):
        return {"homogeneous": True}

    @additive_decorator
    def additive(self, matrix):
        return {"additive": True}

    def status(self, matrix, v1=None, v2=None, c=None):
        try:
            # Check matrix properties
            homo_result = self.homogeneity(matrix)
            add_result = self.additive(matrix)

            # Check linearity if vectors and scalar are provided
            if v1 is not None and v2 is not None and c is not None:
                linear_result = self.is_linear(matrix, v1, v2, c)
                combined = {**homo_result, **add_result, **linear_result}
                return {
                    "matrix": matrix.tolist(),
                    "vectors": {"v1": v1.tolist(), "v2": v2.tolist()} if v1 is not None else None,
                    "scalar": c,
                    "status": combined
                }
            else:
                combined = {**homo_result, **add_result}
                return {
                    "matrix": matrix.tolist(),
                    "status": combined
                }
        except ValueError as e:
            return {
                "matrix": matrix.tolist(),
                "vectors": {"v1": v1.tolist(), "v2": v2.tolist()} if v1 is not None else None,
                "scalar": c,
                "error": str(e)
            }

def get_matrix_from_user():
    print("\n--- Enter your matrix ---")
    print("Hint: Enter each row as space-separated numbers, e.g., '1 1' for a row with two 1s.")
    print("Press Enter after each row. Enter 'done' when finished.")

    matrix = []
    while True:
        row_input = input("Enter a row (or 'done' to finish): ").strip()
        if row_input.lower() == 'done':
            break
        try:
            row = [float(num) for num in row_input.split()]
            matrix.append(row)
        except ValueError:
            print("Invalid input. Please enter numbers only, separated by spaces.")
    return np.array(matrix)

def get_vectors_from_user(dim):
    print(f"\n--- Enter two vectors of dimension {dim} ---")
    v1 = input(f"Enter vector v1 (space-separated, {dim} numbers): ").strip()
    v2 = input(f"Enter vector v2 (space-separated, {dim} numbers): ").strip()
    try:
        v1 = np.array([float(num) for num in v1.split()])
        v2 = np.array([float(num) for num in v2.split()])
        if len(v1) != dim or len(v2) != dim:
            raise ValueError(f"Vectors must have {dim} elements.")
        return v1, v2
    except ValueError as e:
        print(f"Invalid input: {e}")
        return get_vectors_from_user(dim)

def get_scalar_from_user():
    try:
        c = float(input("\nEnter a scalar value (e.g., 2): ").strip())
        return c
    except ValueError:
        print("Invalid input. Please enter a number.")
        return get_scalar_from_user()

def save_json_to_file(data, filename="matrix_result.json"):
    try:
        with open(filename, "r") as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    existing_data.append(data)

    with open(filename, "w") as file:
        json.dump(existing_data, file, indent=4)
    print(f"\nResult appended to {filename}")

def main():
    verification = matrixVerification()

    # Get matrix
    matrix = get_matrix_from_user()
    print(f"\nThis is the matrix you entered:\n{matrix}")

    # Check matrix properties
    matrix_status = verification.status(matrix)
    print("\nMatrix properties status:")
    print(json.dumps(matrix_status, indent=4))

    # Ask if user wants to check linearity
    check_linearity = input("\nWould you like to check linearity with vectors? (yes/no): ").strip().lower()
    if check_linearity == "yes":
        dim = matrix.shape[1]
        v1, v2 = get_vectors_from_user(dim)
        c = get_scalar_from_user()

        # Check linearity
        linearity_status = verification.status(matrix, v1, v2, c)
        print("\nLinearity status:")
        print(json.dumps(linearity_status, indent=4))

        # Save to JSON
        save_choice = input("\nWould you like to save this result to a JSON file? (yes/no): ").strip().lower()
        if save_choice == "yes":
            filename = input("Enter the filename (default: matrix_result.json): ").strip()
            if not filename:
                filename = "matrix_result.json"
            save_json_to_file(linearity_status, filename)

    else:
        # Save matrix properties to JSON
        save_choice = input("\nWould you like to save the matrix properties to a JSON file? (yes/no): ").strip().lower()
        if save_choice == "yes":
            filename = input("Enter the filename (default: matrix_result.json): ").strip()
            if not filename:
                filename = "matrix_result.json"
            save_json_to_file(matrix_status, filename)
        else:
            print("\nResult not saved.")

if __name__ == "__main__":
    main()
