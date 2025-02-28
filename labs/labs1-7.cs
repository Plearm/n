public void DisplayShortInfo()
{
    Console.WriteLine($"Компания: {Name}, Телефон: {Phone}, Контактное лицо: {ContactPerson}");
}


DisplayShortInfo заменить на метод который выводит на экран и почитать по нему информацию



public override string ToString()
{
    return $"Компания: {Name}, Телефон: {Phone}, Контактное лицо: {ContactPerson}";
}
