

dataset = int(
    input(
        f"Choose a dataset (Enter an interger) -"
        f" \n'1' for Berlin52 or"
        f"\n'2' for eli101 or "
        f"\n'3' for a280\n "
    )
)

print(f"You selected {dataset}")

# distance_matrix=" "

if dataset == 1:
    from data.tsp_data_all import distance_matrix_Berlin52 as distance_matrix
    print(len(distance_matrix))
elif dataset == 2:
    from data.tsp_data_all import distance_matrix_eil101 as distance_matrix
    print(len(distance_matrix))
elif dataset == 3:
    from data.tsp_data_all import distance_matrix_a280 as distance_matrix
    print(len(distance_matrix))
else: 
    print("invalid input")


