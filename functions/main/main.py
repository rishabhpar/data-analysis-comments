def hello_get(request):
	first_name = request.args.get('first_name')
    last_name = request.args.get('last_name')
	return 'Hello, ' + first_name + " " + last_name
