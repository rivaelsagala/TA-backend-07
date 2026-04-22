from datetime import datetime

# app/utils.py
def response(data=None, pagination_info=None, status="success", message="Request successful", errors=None):
    if errors is not None:
        return {
            "status": "error",
            "message": "An error occurred",
            "errors": errors,
        }
    
    if pagination_info is None:
        return {
            "status": status,
            "message": message,
            "data": data,
        }
    
    return {
        "status": status,       
        "message": message,        
        "page": pagination_info.page,
        "page_size": pagination_info.per_page,
        "pages": pagination_info.pages,
        "total": pagination_info.total,
        "data": data,        
    }

def formatDate(date_string, from_format, to_format):
    date_object = datetime.strptime(date_string, from_format)
    formatted_date = date_object.strftime(to_format)
    return formatted_date