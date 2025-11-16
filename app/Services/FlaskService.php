<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;

class FlaskService
{
    public function predict($message)
    {
        $response = Http::asJson()->post(env('FLASK_API_URL') . '/predict', [
    'message' => $message,
     ]);
 

        return $response->json();
    }
    public function generate(string|array $texts)
    {
        $payload = is_array($texts) ? ['texts' => $texts] : ['texts' => [$texts]];

        $response = Http::asJson()->post($this->baseUrl . '/generate', $payload);

        return $response->json();
    }
    public function summarizeTopics(array $texts, int $topN = 5, array $ngram = [1, 2], int $maxLen = 200, int $minLen = 80)
    {
        $payload = [
            'texts' => $texts,
            'top_n' => $topN,
            'keyphrase_ngram_range' => $ngram,
            'max_len' => $maxLen,
            'min_len' => $minLen,
        ];

        $response = Http::asJson()->post($this->baseUrl . '/summarize_topics', $payload);

        return $response->json();
    }

}
