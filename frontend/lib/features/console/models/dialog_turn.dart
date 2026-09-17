class DialogTurn {
  const DialogTurn({
    required this.role,
    required this.text,
    required this.time,
  });

  final String role; // user | assistant
  final String text;
  final String time;

  bool get isUser => role == 'user';
}
