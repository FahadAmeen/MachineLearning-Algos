import xlsxwriter

def writeArray(y_test,y_pred,colNum):
    workbook = xlsxwriter.Workbook('hello.xlsx')
    worksheet = workbook.add_worksheet()
    cell_format = workbook.add_format({'bold': True, 'font_color': 'red'})
    column = colNum
    row = 1
    wrong_pred = 0
    for i, j in zip(y_pred, y_test):
        if i != j:
            worksheet.write(row, column, i, cell_format)
            row += 1
            wrong_pred += 1
        else:
            worksheet.write(row, column, i)
            row += 1
    worksheet.write(row, column, wrong_pred)

    workbook.close()