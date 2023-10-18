import dearpygui.dearpygui as dpg
import pandas as pd
import math
from utilities import solution_implicit
dpg.create_context()

WIDTH = 1300
HEIGHT = 900

btn_α_id = 0
btn_c_id = 0
btn_eps_id = 0
btn_k_id = 0
btn_R_id = 0
btn_n_id = 0

btn_number_lines = 0

btn_I_id = 0
btn_K_id = 0

btn_t_id = 0
btn_x_id = 0

plt1 = []
plt2 = []
table_id1 = 0
table_id2 = 0

btn_save_table_id = 0

plt_fix_t = list()
plt_fix_x = list()
btn_I_tag = list()
btn_K_tag = list()
analytic_fix_t = list()
analytic_fix_x = list()

ti = [5 + 35*i for i in range(8)]
xi = [i for i in range(0, 7)]

id_window = dpg.generate_uuid()
I = 300 
K = 300
[x_list, time_list, u, ht, hx] = solution_implicit(I, K, 0.002, 1.65, 6, 250, 0.59, 0.1, 0.00001)


def analytical(t, x, analytic_fix_t_value, analytic_fix_x_value, x_list, time_list):
    dpg.set_value(analytic_fix_t[0], [x_list, analytic_fix_t_value])
    dpg.set_item_label(analytic_fix_t[0], label="analytic:t=" + str(t) + "c")

    dpg.set_value(analytic_fix_x[0], [time_list, analytic_fix_x_value])
    dpg.set_item_label(analytic_fix_x[0], label="analytic:y=" + str(x) + "cm")

def implicit_for_two_graphic(u, x_list, time_list, ht, hx, K, I):
    count = 0
    for i in ti:
        line = list()
        for j in range(I+1):
            line.append(u[j][round(i/ht)])
        dpg.set_value(plt1[count], [x_list, line])
        dpg.set_item_label(plt1[count], label="t=" + str(i) + " c")
        count = count + 1
        
    count = 0
    for i in xi:
        line = list()
        for j in range(K+1):
            line.append(u[round(i/hx)][j])
        dpg.set_value(plt2[count], [time_list, line])
        dpg.set_item_label(plt2[count], label="y=" + str(i) + " cm")
        count = count + 1
        
def send_items():
    n = dpg.get_value(btn_n_id)
    α = dpg.get_value(btn_α_id)
    eps = math.pow(10, dpg.get_value(btn_eps_id))
    c = dpg.get_value(btn_c_id)
    k = dpg.get_value(btn_k_id)
    R = dpg.get_value(btn_R_id)

    I = dpg.get_value(btn_I_id)
    K = dpg.get_value(btn_K_id)

    l = 6
    T = 250

    t = dpg.get_value(btn_t_id)
    x = dpg.get_value(btn_x_id)

    try:
        [x_list, time_list, u, ht, hx] = solution_implicit(I, K, α, c, l, T, k, R, eps)
    except:
        return
    implicit_for_two_graphic(u, x_list, time_list, ht, hx, K, I)
    send_items_of_lines()

def send_items_of_lines():
    α = dpg.get_value(btn_α_id)
    eps = math.pow(10, dpg.get_value(btn_eps_id))
    c = dpg.get_value(btn_c_id)
    k = dpg.get_value(btn_k_id)
    R = dpg.get_value(btn_R_id)

    number = dpg.get_value(btn_number_lines)
    l = 6
    T = 250
    t = dpg.get_value(btn_t_id)
    x = dpg.get_value(btn_x_id)

    for i in range(number):
        I = dpg.get_value(btn_I_tag[i])
        K = dpg.get_value(btn_K_tag[i])

        [x_list, time_list, u, ht, hx] = solution_implicit(I, K, α, c, l, T, k, R, eps)

        line = list()
        for j in range(I+1):
                line.append(u[j][round(t/ht)]) 
        dpg.set_value(plt_fix_t[i], [x_list, line])
        dpg.set_item_label(plt_fix_t[i], label="implicit schema:t=" + str(t) + f"c I={I} K={K}")

        line = list()
        for j in range(K+1):
                line.append(u[round(x/hx)][j]) 
        dpg.set_value(plt_fix_x[i], [time_list, line])
        dpg.set_item_label(plt_fix_x[i], label="implicit schema:y=" + str(x) + f"cm I={I} K={K}")

def send_number_of_lines():
    size = len(btn_I_tag)
    number = dpg.get_value(btn_number_lines)

    if number < size:
        for j in range(size-1, number-1, -1):
            dpg.delete_item(btn_I_tag[j])
            btn_I_tag.remove(btn_I_tag[j])

            dpg.delete_item(btn_K_tag[j])
            btn_K_tag.remove(btn_K_tag[j])

            dpg.delete_item(plt_fix_t[j])
            plt_fix_t.remove(plt_fix_t[j])

            dpg.delete_item(plt_fix_x[j])
            plt_fix_x.remove(plt_fix_x[j])

    elif number > size: 
        for i in range(size, dpg.get_value(btn_number_lines)):
            btn_I_tag.append(dpg.generate_uuid())
            dpg.set_item_callback(dpg.add_input_int(label="I"+str(i), enabled=True, default_value=300, tag=btn_I_tag[i], parent=id_window), callback=send_items_of_lines)

            btn_K_tag.append(dpg.generate_uuid())
            dpg.set_item_callback(dpg.add_input_int(label="K"+str(i), enabled=True, default_value=300, tag=btn_K_tag[i], parent=id_window), callback=send_items_of_lines)

            I = dpg.get_value(btn_I_tag[i])
            K = dpg.get_value(btn_K_tag[i])

            line = list()
            plt_fix_t.append(dpg.generate_uuid())
            for j in range(I+1):
                    line.append(u[j][round(dpg.get_value(btn_t_id)/ht)])  
            dpg.add_line_series(x_list, line, label=f"implicit schema:t{i}=" + str(dpg.get_value(btn_t_id)) + f"c I={I} K={K}", parent="y_axis3", tag=plt_fix_t[i])

            plt_fix_x.append(dpg.generate_uuid())
            line = list()
            for j in range(K+1):
                    line.append(u[round(dpg.get_value(btn_x_id)/hx)][j])  
            dpg.add_line_series(time_list, line, label=f"implicit schema:y{i}=" + str(dpg.get_value(btn_x_id)) + f"cm I={I} K={K}", parent="y_axis4", tag=plt_fix_x[i])


with dpg.window(label="General params", pos=(0, 0), height=round(HEIGHT / 2 - 20), width=(WIDTH / 2 - 120)) as win:
    btn_eps_id = dpg.add_input_int(label="eps 10^(-x)", enabled=True, default_value=-5)
    btn_α_id = dpg.add_input_double(label="a", enabled=True, default_value=0.002, format="%.4f", step=0.001)
    btn_c_id = dpg.add_input_double(label="c", enabled=True, default_value=1.65, format="%.4f", step=0.001)
    btn_k_id = dpg.add_input_double(label="k", enabled=True, default_value=0.59, format="%.4f", step=0.001)
    btn_R_id = dpg.add_input_double(label="R", enabled=True, default_value=0.1, format="%.4f", step=0.001)
    btn_n_id = dpg.add_input_int(label="n", enabled=True, default_value=100)

    btn_I_id = dpg.add_input_int(label="I general", enabled=True, default_value=300)
    btn_K_id = dpg.add_input_int(label="K general", enabled=True, default_value=300)

    btn_t_id = dpg.add_input_int(label="t", enabled=True, default_value=125)
    dpg.set_item_callback(btn_t_id, callback=send_items)

    btn_x_id = dpg.add_input_int(label="y", enabled=True, default_value=6)
    dpg.set_item_callback(btn_x_id, callback=send_items)
    
    dpg.set_item_callback(btn_K_id, send_items)
    dpg.set_item_callback(btn_I_id, send_items)
        
    dpg.set_item_callback(btn_eps_id, send_items)
    dpg.set_item_callback(btn_α_id, send_items)
    dpg.set_item_callback(btn_c_id, send_items)
    dpg.set_item_callback(btn_R_id, send_items)
    dpg.set_item_callback(btn_n_id, send_items)

    with dpg.group(label="Params for lines", tag=id_window):
        dpg.add_text("Add a number of lines", tag="text item")
        btn_number_lines = dpg.add_input_int(label="number of lines", enabled=True, default_value=1)
        btn_I_tag = []
        btn_K_tag = []
        for i in range(dpg.get_value(btn_number_lines)):
            btn_I_tag.append(dpg.generate_uuid())
            btn_K_tag.append(dpg.generate_uuid())
            dpg.set_item_callback(dpg.add_input_int(label="I"+str(i), enabled=True, default_value=300, tag=btn_I_tag[i]), callback=send_items_of_lines)
            dpg.set_item_callback(dpg.add_input_int(label="K"+str(i), enabled=True, default_value=300, tag=btn_K_tag[i]), callback=send_items_of_lines)

        dpg.set_item_callback(btn_number_lines, send_number_of_lines)


with dpg.window(label="PLOT", tag="win", pos=(WIDTH / 2 - 120, 0), width=round(3 * WIDTH / 4 + 30)):
    with dpg.plot(label="plt1", height=round(HEIGHT / 2 - 50), width=round(3 * WIDTH / 4), anti_aliased=True,
                  no_title=True):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="Y, cm", tag="x_axis1")
        dpg.add_plot_axis(dpg.mvYAxis, label="w, K", tag="y_axis1")

        I = dpg.get_value(btn_I_id)
        for i in ti:
            tag = dpg.generate_uuid()
            plt1.append(tag)
            line = list()
            for j in range(I + 1):
                line.append(u[j][round(i/ht)])
            dpg.add_line_series(x_list, line, label="t=" + str(i) + " c", parent="y_axis1", tag=tag)

    with dpg.plot(label="plt2", height=round(HEIGHT / 2 - 50), width=round(3 * WIDTH / 4), anti_aliased=True,
                  no_title=True):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="T, c", tag="x_axis2")
        dpg.add_plot_axis(dpg.mvYAxis, label="w, K", tag="y_axis2")

        K = dpg.get_value(btn_K_id)
        for i in xi:
            tag = dpg.generate_uuid()
            plt2.append(tag)
            line = list()
            for j in range(K+1):
                line.append(u[round(i/hx)][j])
            dpg.add_line_series(time_list, line, label="y=" + str(i) + " cm", parent="y_axis2", tag=tag)

    with dpg.plot(label="plt3", height=round(HEIGHT / 2 - 50), width=round(3 * WIDTH / 4), anti_aliased=True,
                  no_title=True):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="Y, cm", tag="x_axis3")
        dpg.add_plot_axis(dpg.mvYAxis, label="w, K", tag="y_axis3")

        for i in range(dpg.get_value(btn_number_lines)):
            I = dpg.get_value(btn_I_tag[i])
            K = dpg.get_value(btn_K_tag[i])
            line = list()
            plt_fix_t.append(dpg.generate_uuid())
            for j in range(I+1):
                    line.append(u[j][round(dpg.get_value(btn_t_id)/ht)])  
            dpg.add_line_series(x_list, line, label=f"implicit schema:t{i}=" + str(dpg.get_value(btn_t_id)) + f"c I={I} K={K}", parent="y_axis3", tag=plt_fix_t[i])

    with dpg.plot(label="plt4", height=round(HEIGHT / 2 - 50), width=round(3 * WIDTH / 4), anti_aliased=True,
                  no_title=True):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis,  label="T, c", tag="x_axis4")
        dpg.add_plot_axis(dpg.mvYAxis, label="w, K", tag="y_axis4")
        
        for i in range(dpg.get_value(btn_number_lines)):
            plt_fix_x.append(dpg.generate_uuid())
            K = dpg.get_value(btn_K_tag[i])
            I = dpg.get_value(btn_I_tag[i])
            line = list()
            for j in range(K+1):
                    line.append(u[round(dpg.get_value(btn_x_id)/hx)][j])  
            dpg.add_line_series(time_list, line, label=f"implicit schema:y{i}=" + str(dpg.get_value(btn_x_id)) + f"cm I={I} K={K}", parent="y_axis4", tag=plt_fix_x[i])


def find_w_in_node_I_K(I_node, K_node, N1, N2):
    """
        N1: ht/N1;
        N2: hy/N2;
    """
    α = dpg.get_value(btn_α_id)
    eps = math.pow(10, dpg.get_value(btn_eps_id))
    c = dpg.get_value(btn_c_id)
    k = dpg.get_value(btn_k_id)
    R = dpg.get_value(btn_R_id)
    
    l = 6
    T = 125
    
    [_, _, u, _, _] = solution_implicit(I_node*N1, K_node*N2, α, c, l, T, k, R, eps)
    return abs(u[round(I_node*N1/2)][round(K_node*N2/2)])



def solve_delta():
    global delta_row, value_row
    keys = list(value_row.keys())
    for j in keys: 
        val = list()
        for i in range(len(value_row[j])-1):
            val.append(value_row[j][i] - value_row[j][i+1])
        val1 = list()
        for i in range(len(val) - 1):
            val1.append(abs(val[i]/val[i+1]))
        delta_row[j] = val1
    return delta_row

value_row = {}
row = 0
delta_row = {}

def create_table_two(table):
    delta_row = solve_delta()
    for j in range(0, len(table_values["I"])):
        with dpg.table_row(parent=table):
                    dpg.add_button(label=f"{table_values['I'][j]}") 
                    dpg.add_button(label=f"{table_values['K'][j]}") 
                    dpg.add_button(label=f"{delta_row[j][0]}") 

table_columns1 = list()
def create_table_one(table, unexist=True):
    global table_columns1, row, value_row
    if unexist:
        table_columns1.append(dpg.add_table_column(label="I"))
        table_columns1.append(dpg.add_table_column(label="K"))
        table_columns1.append(dpg.add_table_column(label="w_ht_hx"))

    I_node = dpg.get_value(btn_table_I_id)
    K_node = dpg.get_value(btn_table_K_id)

    table_values["I"].append(I_node)
    table_values["K"].append(K_node)
    table_values["w_ht_hx"].append(find_w_in_node_I_K(I_node, K_node, 1, 1))
    table_columns1 = add_eps_columns(table_columns1, I_node, K_node, unexist)


    table_id[table_id1] = table_columns1
    keys = list(table_values.keys())
    for j in range(0, len(table_values["I"])):
        with dpg.table_row(parent=table, tag=f"table1_{j}"):
            value = list()
            for i in range(0, len(table_columns1)):
                with dpg.table_cell():
                    if "w" in keys[i]:
                        value.append(table_values[keys[i]][j])
                    dpg.add_button(label=f"{table_values[keys[i]][j]}")
    value_row[row] = value
    row += 1

def clear_table():
    I_node = dpg.get_value(btn_table_I_id)
    k_node = dpg.get_value(btn_table_K_id)
    ids = table_id.keys()
    for j in ids:
        for tag in dpg.get_item_children(j)[1]:
            dpg.delete_item(tag)

    create_table_one(list(ids)[0], False)
    create_table_two(list(ids)[1])


def add_eps_columns(table_columns:list, I_node, K_node, unexist=True):
    i = 4
    while i <= 16:
        if unexist:
            table_columns.append(dpg.add_table_column(label=f"w_ht/{i}_hx/{int(math.sqrt(i))}"))
            table_values[f"w_ht/{i}_hx/{str(math.sqrt(i))}"] = []
        table_values[f"w_ht/{i}_hx/{str(math.sqrt(i))}"].append(find_w_in_node_I_K(I_node, K_node, i, int(math.sqrt(i))))
        i = i**2
    return table_columns

table_id = {}
with dpg.window(label="Table", pos=(0, HEIGHT / 2 - 20)) as win:
    table_values = {'I': [], 'K': [], 'w_ht_hx': []}

    btn_table_I_id = dpg.add_input_int(label="I node", enabled=True, default_value=10, step=1)
    btn_table_K_id = dpg.add_input_int(label="K node", enabled=True, default_value=40, step=1)
    I_node = dpg.get_value(btn_table_I_id)
    K_node = dpg.get_value(btn_table_K_id)

    btn_delta_id = dpg.add_button(label="calculte")
    dpg.set_item_callback(btn_delta_id, solve_delta)
    dpg.set_item_callback(btn_delta_id, clear_table)
    table_id1= dpg.generate_uuid()
    with dpg.table(header_row=True, width=WIDTH / 2 - 135, height=HEIGHT/6-20, tag=table_id1):
        create_table_one(table_id1)

    table_id2 = dpg.generate_uuid()
    with dpg.table(header_row=True, width=WIDTH / 2 - 135, height=HEIGHT/6-20, tag=table_id2):
        value = solve_delta()
        table_columns2 = list()
        table_columns2.append(dpg.add_table_column(label="I"))
        table_columns2.append(dpg.add_table_column(label="K"))

        for i in range(len(value)):
            table_columns2.append(dpg.add_table_column(label=f"delta{i}"))

        table_id[table_id2] = table_columns2
        create_table_two(table_id2)

dpg.create_viewport(title='Modeling thermal process', width=WIDTH, height=HEIGHT)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
