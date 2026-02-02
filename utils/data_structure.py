class RunResult:
    def __init__(
            self,
            compilation_error=True,
            execution_error=True,
            output=None,
            correct=False,
            execution_time=1e12,
            execution_time_type="instructions",
            time_limit_error=False
    ):
        self.compilation_error = compilation_error
        self.execution_error = execution_error
        self.output = output
        self.correct = correct
        self.execution_time = execution_time
        self.execution_time_type = execution_time_type
        self.time_limit_error = time_limit_error

    @classmethod
    def from_json(cls, json_obj):
        return cls(
            compilation_error=json_obj.get("compilation_error", True),
            execution_error=json_obj.get("execution_error", True),
            output=json_obj.get("output"),
            correct=json_obj.get("correct", False),
            execution_time=json_obj.get("execution_time", 1e12),
            execution_time_type=json_obj.get("execution_time_type", "instructions"),
            time_limit_error=json_obj.get("time_limit_error", False)
        )

    @classmethod
    def from_dict(cls, dict_obj):
        return cls(
            compilation_error=dict_obj.get("compilation_error", True),
            execution_error=dict_obj.get("execution_error", True),
            output=dict_obj.get("output"),
            correct=dict_obj.get("correct", False),
            execution_time=dict_obj.get("execution_time", 1e12),
            execution_time_type=dict_obj.get("execution_time_type", "instructions"),
            time_limit_error=dict_obj.get("time_limit_error", False)
        )

    def to_dict(self):
        # Ensure all fields are serializable
        def safe(val):
            if isinstance(val, bytes):
                return val.decode('utf-8', errors='replace')
            if isinstance(val, (list, dict, str, int, float, bool)) or val is None:
                return val
            return str(val)
        return {
            "compilation_error": safe(self.compilation_error),
            "execution_error": safe(self.execution_error),
            "output": safe(self.output),
            "correct": safe(self.correct),
            "execution_time": safe(self.execution_time),
            "execution_time_type": safe(self.execution_time_type),
            "time_limit_error": safe(self.time_limit_error)
        }

    def to_json(self):
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def set_compilation_error(self, error_message):
        self.compilation_error = error_message

    def set_execution_error(self, error_message):
        self.execution_error = error_message

    def set_output(self, output):
        self.output = output

    def set_correct(self, correct):
        self.correct = correct

    def set_execution_time(self, time_taken):
        self.execution_time = time_taken

    def __str__(self):
        return f"Compilation Error: {self.compilation_error}\nExecution Error: {self.execution_error}\nOutput: {self.output}\nCorrect: {self.correct}\nExecution Time: {self.execution_time}"
