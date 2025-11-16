<?php

namespace App\Http\Controllers;

use App\Services\FlaskService;
use Illuminate\Http\Request;

class AIController extends Controller
{
    protected $flask;

    public function __construct(FlaskService $flask)
    {
        $this->flask = $flask;
    }

    public function chat(Request $request)
    {
        $userMessage = $request->input('message');
        $response = $this->flask->predict($userMessage);

        return response()->json($response);
    }
}
